"""Scoreformer: performed notes -> score tokens, with time-aware attention.

Same interface as ``src.model.Score2Seq`` (``forward(notes, note_mask, dec_in)`` and
``generate(notes, note_mask, prefix, eos_id)``), so it trains on the same batches and is scored by
the same evaluation code. What is different:

Encoder (one vector per performed note, input = the Dataset's [B, N, 8] features)
  * The standardized features are mapped back to seconds inside the model (the train-set mean/std
    are stored in the config), so the encoding does not depend on the feature scaling.
  * Inter-onset gap and duration are log-binned and embedded, velocity is binned: an embedding per
    bin can learn non-linear effects ("a 30 ms gap means a chord, 300 ms means a new beat") that a
    single linear layer on the standardized value cannot. The standardized values are still added
    through a linear layer for the detail inside a bin.
  * **Time rotary attention**: self-attention uses rotary position embeddings driven by each note's
    onset time in seconds instead of its index, with periods from 20 ms to 60 s. Attention between
    two notes then depends on how far apart in time they were played, at any resolution a
    rhythm needs; the note index carries no musical meaning (chord notes are ordered by pitch).
Decoder
  * Standard rotary positions over the token index (no learned position table, so no hard length
    limit), causal self-attention + cross-attention to the notes, SwiGLU feed-forward, pre-norm.
  * Tied input/output embeddings; KV-cached greedy decoding.

Sizes (``PRESETS``): small ~8M, base ~45M, large ~133M parameters (vocab 621).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "small": dict(d_model=256, n_heads=4, enc_layers=4, dec_layers=4, ff=1024),
    "base": dict(d_model=512, n_heads=8, enc_layers=6, dec_layers=6, ff=2048),
    "large": dict(d_model=768, n_heads=12, enc_layers=8, dec_layers=8, ff=3072),
}


@dataclass
class ScoreformerConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    ff: int = 2048
    dropout: float = 0.1
    max_tokens: int = 2048        # longest decoder sequence (prompt + target) generate() will produce
    max_notes: int = 1024         # inputs are truncated to this many notes by the collate
    # train-set mean / std of the standardized feature columns 1-4 (onset, ioi, log_dur, velocity);
    # identity if the Dataset doesn't standardize
    feat_mean: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    feat_std: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    time_min_period: float = 0.02  # seconds, finest time scale of the encoder's rotary attention
    time_max_period: float = 60.0
    n_time_bins: int = 64
    n_vel_bins: int = 32

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------

def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels of x [B,H,T,dh] by angles given as cos/sin [B,1,T,dh/2]."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def rope_angles(pos: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pos [B,T] (index or seconds) -> cos, sin [B,1,T,dh/2]. Computed in float32 (onset seconds x
    2pi/20ms reach ~20k rad, far beyond bf16 precision); ``rope`` casts the result."""
    ang = pos.float()[..., None] * inv_freq.float()
    return ang.cos()[:, None], ang.sin()[:, None]


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int, p: float):
        super().__init__()
        h = int(ff * 2 / 3)  # same parameter count as a GELU MLP of width ff
        self.w12 = nn.Linear(d, 2 * h)
        self.w3 = nn.Linear(h, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(a) * b))


class Attention(nn.Module):
    """Multi-head attention on F.scaled_dot_product_attention. Self-attention takes rotary cos/sin
    and an optional KV cache; cross-attention takes precomputed memory keys/values."""

    def __init__(self, d: int, h: int, p: float, cross: bool = False):
        super().__init__()
        self.h, self.dh, self.p = h, d // h, p
        if cross:
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(d, 2 * d)
        else:
            self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def _merge(self, x):
        B, H, T, dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * dh)

    def memory_kv(self, memory):
        k, v = self.kv(memory).chunk(2, dim=-1)
        return self._split(k), self._split(v)

    def self_attn(self, x, cos, sin, mask=None, causal=False, cache=None):
        q, k, v = (self._split(t) for t in self.qkv(x).chunk(3, dim=-1))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        p = self.p if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p,
                                           is_causal=causal and mask is None)
        return self.o(self._merge(a)), (k, v)

    def cross_attn(self, x, mem_kv, mem_mask):
        p = self.p if self.training else 0.0
        a = F.scaled_dot_product_attention(self._split(self.q(x)), *mem_kv, attn_mask=mem_mask, dropout_p=p)
        return self.o(self._merge(a))


class EncoderLayer(nn.Module):
    def __init__(self, c: ScoreformerConfig):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = Attention(c.d_model, c.n_heads, c.dropout)
        self.ff = SwiGLU(c.d_model, c.ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, mask):
        x = x + self.drop(self.attn.self_attn(self.n1(x), cos, sin, mask=mask)[0])
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, c: ScoreformerConfig):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(c.d_model) for _ in range(3))
        self.self_attn = Attention(c.d_model, c.n_heads, c.dropout)
        self.cross = Attention(c.d_model, c.n_heads, c.dropout, cross=True)
        self.ff = SwiGLU(c.d_model, c.ff, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, mem_kv, mem_mask, self_mask=None, cache=None):
        a, kv = self.self_attn.self_attn(self.n1(x), cos, sin, mask=self_mask, causal=self_mask is None,
                                         cache=cache)
        x = x + self.drop(a)
        x = x + self.drop(self.cross.cross_attn(self.n2(x), mem_kv, mem_mask))
        return x + self.drop(self.ff(self.n3(x))), kv


# ---------------------------------------------------------------------------------------------
# encoder input
# ---------------------------------------------------------------------------------------------

def log_bin(x: torch.Tensor, lo: float, hi: float, n: int) -> torch.Tensor:
    """Bin non-negative seconds on a log scale: 0 -> bin 0, ``hi`` and above -> bin n-1."""
    x = x.clamp(min=0)
    return (torch.log1p(x / lo) / math.log1p(hi / lo) * (n - 1)).round().clamp(0, n - 1).long()


class NoteEmbedding(nn.Module):
    """[B,N,8] Dataset features -> [B,N,D] note vectors, plus each note's onset in seconds."""

    def __init__(self, c: ScoreformerConfig):
        super().__init__()
        d = c.d_model
        self.pitch = nn.Embedding(89, d, padding_idx=0)       # 0 = padding, 1..88 = keys
        self.ioi = nn.Embedding(c.n_time_bins, d)
        self.dur = nn.Embedding(c.n_time_bins, d)
        self.vel = nn.Embedding(c.n_vel_bins, d)
        self.cont = nn.Linear(4, d)                           # standardized onset, ioi, log dur, vel
        self.flags = nn.Linear(3, d)                          # pedal at onset / release, carried over
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(c.dropout)
        self.n_time, self.n_vel = c.n_time_bins, c.n_vel_bins
        self.register_buffer("mean", torch.tensor(c.feat_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("std", torch.tensor(c.feat_std, dtype=torch.float32), persistent=False)

    def forward(self, x, mask):
        raw = x[..., 1:5].float() * self.std + self.mean      # onset s, ioi s, log dur, velocity/127
        onset, ioi, dur, vel = raw[..., 0], raw[..., 1], raw[..., 2].exp(), raw[..., 3]
        pitch = torch.where(mask, (x[..., 0] - 20).clamp(1, 88), torch.zeros_like(x[..., 0])).long()
        h = (self.pitch(pitch)
             + self.ioi(log_bin(ioi, 0.005, 4.0, self.n_time))
             + self.dur(log_bin(dur, 0.01, 10.0, self.n_time))
             + self.vel((vel * self.n_vel).long().clamp(0, self.n_vel - 1))
             + self.cont(x[..., 1:5]) + self.flags(x[..., 5:8]))
        h = h * mask[..., None]
        return self.drop(self.norm(h)), torch.where(mask, onset, torch.zeros_like(onset))


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------

class Scoreformer(nn.Module):
    arch = "scoreformer"

    def __init__(self, c: ScoreformerConfig):
        super().__init__()
        self.cfg = c
        dh = c.d_model // c.n_heads
        assert dh % 2 == 0 and c.d_model % c.n_heads == 0
        self.embed = NoteEmbedding(c)
        self.enc = nn.ModuleList(EncoderLayer(c) for _ in range(c.enc_layers))
        self.enc_norm = nn.LayerNorm(c.d_model)
        self.tok = nn.Embedding(c.vocab_size, c.d_model, padding_idx=0)
        self.tok_drop = nn.Dropout(c.dropout)
        self.dec = nn.ModuleList(DecoderLayer(c) for _ in range(c.dec_layers))
        self.out_norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        # rotary frequencies: encoder = onset seconds (periods time_min..time_max, geometric),
        # decoder = token index (the usual base 10000)
        periods = torch.logspace(math.log10(c.time_min_period), math.log10(c.time_max_period), dh // 2)
        self.register_buffer("time_freq", 2 * math.pi / periods, persistent=False)
        self.register_buffer("tok_freq", 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh)), persistent=False)
        self._init_weights()

    def _init_weights(self):
        n_res = 2 * (self.cfg.enc_layers + self.cfg.dec_layers)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("attn.o.weight", "cross.o.weight", "w3.weight")):
                    p.data.mul_(1 / math.sqrt(n_res))  # GPT-2 residual scaling
            elif name.endswith(".bias"):
                nn.init.zeros_(p)
        with torch.no_grad():
            self.tok.weight[0].zero_()
            self.embed.pitch.weight[0].zero_()

    # ---- encoder
    def encode(self, notes, note_mask):
        h, onset = self.embed(notes, note_mask)
        cos, sin = rope_angles(onset, self.time_freq)
        mask = note_mask[:, None, None, :]
        for layer in self.enc:
            h = layer(h, cos, sin, mask)
        return self.enc_norm(h), mask

    # ---- training
    def forward(self, notes, note_mask, dec_in):
        memory, mm = self.encode(notes, note_mask)
        T = dec_in.shape[1]
        pos = torch.arange(T, device=dec_in.device)[None].expand(dec_in.shape[0], T)
        h = self.tok_drop(self.tok(dec_in))
        cos, sin = rope_angles(pos, self.tok_freq)
        for layer in self.dec:
            h, _ = layer(h, cos, sin, layer.cross.memory_kv(memory), mm)
        return self.head(self.out_norm(h))

    # ---- inference
    @torch.no_grad()
    def generate(self, notes, note_mask, prefix: list[torch.Tensor], eos_id: int, max_new: int = 1200):
        """Greedy decoding with a KV cache. ``prefix[b]`` = <bos> [prompt <sep>] for item b.
        Prefixes are left-padded; padded positions are masked as keys and rotary positions count
        real tokens only. Returns, per item, the generated ids after the prefix (without <eos>)."""
        B, dev = len(prefix), notes.device
        memory, mm = self.encode(notes, note_mask)
        mem_kv = [layer.cross.memory_kv(memory) for layer in self.dec]
        Lp = max(len(p) for p in prefix)
        seq = torch.zeros(B, Lp, dtype=torch.long, device=dev)
        valid = torch.zeros(B, Lp, dtype=torch.bool, device=dev)
        for b, p in enumerate(prefix):
            seq[b, Lp - len(p):] = p.to(dev)
            valid[b, Lp - len(p):] = True
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        m = torch.ones(Lp, Lp, dtype=torch.bool, device=dev).tril()[None] & valid[:, None, :]
        m |= torch.eye(Lp, dtype=torch.bool, device=dev)[None]  # padded rows attend to themselves
        h = self.tok(seq)
        cos, sin = rope_angles(pos, self.tok_freq)
        caches = []
        for layer, kv in zip(self.dec, mem_kv):
            h, c = layer(h, cos, sin, kv, mm, self_mask=m[:, None])
            caches.append(c)
        logits = self.head(self.out_norm(h[:, -1]))
        max_new = max(0, min(max_new, self.cfg.max_tokens - int(valid.sum(1).max())))
        out = torch.zeros(B, max_new, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        cur = pos[:, -1]
        n_out = 0
        for step in range(max_new):
            nxt = torch.where(done, torch.zeros_like(done, dtype=torch.long), logits.argmax(-1))
            out[:, step] = nxt
            n_out = step + 1
            done |= nxt == eos_id
            if done.all():
                break
            valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
            cur = cur + 1
            h = self.tok(nxt[:, None])
            cos, sin = rope_angles(cur[:, None], self.tok_freq)
            sm = valid[:, None, None, :]
            for i, (layer, kv) in enumerate(zip(self.dec, mem_kv)):
                h, caches[i] = layer(h, cos, sin, kv, mm, self_mask=sm, cache=caches[i])
            logits = self.head(self.out_norm(h[:, -1]))
        res = []
        for b in range(B):
            g = out[b, :n_out].tolist()
            res.append(g[:g.index(eos_id)] if eos_id in g else g)
        return res

    @torch.no_grad()
    def generate_naive(self, notes, note_mask, prefix: list[torch.Tensor], eos_id: int, max_new: int = 1200):
        """Reference greedy decoding without a cache, one item at a time (for tests)."""
        res = []
        for b, p in enumerate(prefix):
            seq = p.to(notes.device)[None]
            gen = []
            for _ in range(min(max_new, self.cfg.max_tokens - len(p))):
                nxt = int(self(notes[b:b + 1], note_mask[b:b + 1], seq)[0, -1].argmax())
                if nxt == eos_id:
                    break
                gen.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], 1)
            res.append(gen)
        return res


# ---------------------------------------------------------------------------------------------
# checkpoints (both architectures)
# ---------------------------------------------------------------------------------------------

def build_model(arch: str, cfg: dict) -> nn.Module:
    if arch == "scoreformer":
        return Scoreformer(ScoreformerConfig(**cfg))
    from .model import ModelConfig, Score2Seq
    return Score2Seq(ModelConfig(**cfg))


def load_model(path, device) -> nn.Module:
    """Load a checkpoint from scripts/train_model.py (either arch) or scripts/train.py (Score2Seq)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck.get("arch", "score2seq"), ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()
