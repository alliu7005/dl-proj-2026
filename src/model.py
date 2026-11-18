"""Encoder-decoder transformer: performed notes -> score tokens.

Encoder input: the Dataset's per-note features [B, N, 8] (see ``dataset.note_features``):
    0 pitch (MIDI, raw)          -> embedding (88 keys + padding)
    1 onset (standardized)       -> Fourier features (time position) + linear
    2-7 ioi, log duration, velocity (standardized), pedal at onset / release, carried-over (0/1)
                                 -> linear
plus a sinusoidal encoding of the note's index in the window (notes are sorted by onset).

Decoder: token embedding + learned positions, pre-norm layers with causal self-attention and
cross-attention to the notes, written directly on ``F.scaled_dot_product_attention``:
  * training: right-padded targets with a pure causal mask (padding is always after the real
    tokens, so causality already hides it from every position the loss uses), which lets PyTorch
    use its fused flash / memory-efficient kernels;
  * generation: a key/value cache (one new token per step attends to cached keys) and the
    cross-attention keys/values of the notes computed once.
Teacher forcing: feed ``target[:, :-1]``, predict ``target[:, 1:]`` where ``loss_mask[:, 1:]``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 6
    ff: int = 1024
    dropout: float = 0.1
    max_tokens: int = 2048       # decoder positions (prompt + target)
    max_notes: int = 1024        # encoder positions (inputs are truncated to this)
    n_fourier: int = 16          # onset Fourier features

    def to_dict(self):
        return asdict(self)


def _sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).float()[:, None]
    i = torch.arange(0, d, 2).float()[None, :]
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class NoteEncoder(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.pitch = nn.Embedding(90, c.d_model, padding_idx=0)       # 0 = padding, 1..88 = keys
        self.cont = nn.Linear(7, c.d_model)
        self.freqs = nn.Parameter(torch.logspace(-1, 1.5, c.n_fourier) * math.pi, requires_grad=False)
        self.fourier = nn.Linear(2 * c.n_fourier, c.d_model)
        self.register_buffer("pos", _sinusoid(c.max_notes, c.d_model), persistent=False)
        self.norm = nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)
        layer = nn.TransformerEncoderLayer(c.d_model, c.n_heads, c.ff, c.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.layers = nn.TransformerEncoder(layer, c.enc_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(c.d_model)

    def forward(self, x, mask):
        """x [B,N,8], mask [B,N] True = real note -> memory [B,N,D]."""
        n = x.shape[1]
        pitch = torch.where(mask, (x[..., 0] - 20).clamp(1, 89), torch.zeros_like(x[..., 0])).long()
        ang = x[..., 1:2] * self.freqs
        h = (self.pitch(pitch) + self.cont(x[..., 1:]) + self.fourier(torch.cat([ang.sin(), ang.cos()], -1))
             + self.pos[:n])
        h = self.drop(self.norm(h))
        return self.out_norm(self.layers(h, src_key_padding_mask=~mask))


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer on F.scaled_dot_product_attention, with an optional KV cache."""

    def __init__(self, d: int, h: int, ff: int, p: float):
        super().__init__()
        self.h, self.dh, self.p = h, d // h, p
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.so = nn.Linear(d, d)
        self.cq = nn.Linear(d, d)
        self.ckv = nn.Linear(d, 2 * d)
        self.co = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def _split(self, x):  # [B,T,D] -> [B,H,T,dh]
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def _merge(self, x):  # [B,H,T,dh] -> [B,T,D]
        B, H, T, dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * dh)

    def cross_kv(self, memory):
        k, v = self.ckv(memory).chunk(2, dim=-1)
        return self._split(k), self._split(v)

    def forward(self, x, mem_kv, mem_mask, self_mask=None, cache=None):
        """x [B,T,D]; mem_kv = cross_kv(memory); mem_mask [B,1,1,N] True = attend.
        Training: self_mask None -> causal. Generation: ``cache`` = (k, v) of earlier positions
        (or None for the prefix pass) and ``self_mask`` [B,1,T,L] True = attend.
        Returns (out, (k, v)) with k/v covering all positions so far."""
        p = self.p if self.training else 0.0
        q, k, v = (self._split(t) for t in self.qkv(self.n1(x)).chunk(3, dim=-1))
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        if self_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=self_mask, dropout_p=p)
        x = x + self.drop(self.so(self._merge(a)))
        a = F.scaled_dot_product_attention(self._split(self.cq(self.n2(x))), *mem_kv, attn_mask=mem_mask,
                                           dropout_p=p)
        x = x + self.drop(self.co(self._merge(a)))
        x = x + self.drop(self.ff(self.n3(x)))
        return x, (k, v)


class Score2Seq(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.encoder = NoteEncoder(c)
        self.tok = nn.Embedding(c.vocab_size, c.d_model, padding_idx=0)
        self.tok_pos = nn.Embedding(c.max_tokens, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList(DecoderLayer(c.d_model, c.n_heads, c.ff, c.dropout)
                                    for _ in range(c.dec_layers))
        self.out_norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        # small init: nn.Embedding's default N(0, 1) makes the tied output logits huge
        for emb in (self.tok, self.tok_pos, self.encoder.pitch):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.tok.weight[0].zero_()
            self.encoder.pitch.weight[0].zero_()

    @staticmethod
    def _mem_mask(note_mask):
        return note_mask[:, None, None, :]  # [B,1,1,N] True = attend

    def forward(self, notes, note_mask, dec_in):
        memory = self.encoder(notes, note_mask)
        mm = self._mem_mask(note_mask)
        t = dec_in.shape[1]
        h = self.drop(self.tok(dec_in) + self.tok_pos(torch.arange(t, device=dec_in.device)))
        for layer in self.layers:
            h, _ = layer(h, layer.cross_kv(memory), mm)
        return self.head(self.out_norm(h))

    @torch.no_grad()
    def generate(self, notes, note_mask, prefix: list[torch.Tensor], eos_id: int, max_new: int = 1200):
        """Greedy decoding with a KV cache. ``prefix[b]`` = <bos> [prompt <sep>] for item b.
        Prefixes are LEFT-padded so every row's next token sits at the same column; padded
        positions are masked out as keys and position ids count real tokens only.
        Returns, per item, the generated token ids after the prefix (without <eos>)."""
        B, dev = len(prefix), notes.device
        memory = self.encoder(notes, note_mask)
        mm = self._mem_mask(note_mask)
        mem_kv = [layer.cross_kv(memory) for layer in self.layers]
        Lp = max(len(p) for p in prefix)
        seq = torch.zeros(B, Lp, dtype=torch.long, device=dev)
        valid = torch.zeros(B, Lp, dtype=torch.bool, device=dev)
        for b, p in enumerate(prefix):
            seq[b, Lp - len(p):] = p.to(dev)
            valid[b, Lp - len(p):] = True
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        # prefix pass: causal & key valid; a padded query attends to itself (avoids empty rows)
        causal = torch.ones(Lp, Lp, dtype=torch.bool, device=dev).tril()
        m = causal[None] & valid[:, None, :]
        m |= torch.eye(Lp, dtype=torch.bool, device=dev)[None]
        h = self.tok(seq) + self.tok_pos(pos)
        caches = []
        for layer, kv in zip(self.layers, mem_kv):
            h, c = layer(h, kv, mm, self_mask=m[:, None])
            caches.append(c)
        logits = self.head(self.out_norm(h[:, -1]))
        max_new = min(max_new, self.cfg.max_tokens - int(valid.sum(1).max()))
        out = torch.zeros(B, max_new, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        n_out = 0
        cur_pos = pos[:, -1]
        for step in range(max_new):
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            out[:, step] = nxt
            n_out = step + 1
            done |= nxt == eos_id
            if done.all():
                break
            valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
            cur_pos = cur_pos + 1
            h = self.tok(nxt[:, None]) + self.tok_pos(cur_pos.clamp(max=self.cfg.max_tokens - 1))[:, None]
            sm = valid[:, None, None, :]
            new_caches = []
            for layer, kv, c in zip(self.layers, mem_kv, caches):
                h, c = layer(h, kv, mm, self_mask=sm, cache=c)
                new_caches.append(c)
            caches = new_caches
            logits = self.head(self.out_norm(h[:, -1]))
        res = []
        for b in range(B):
            g = out[b, :n_out].tolist()
            res.append(g[:g.index(eos_id)] if eos_id in g else g)
        return res

    @torch.no_grad()
    def generate_naive(self, notes, note_mask, prefix: list[torch.Tensor], eos_id: int, max_new: int = 1200):
        """Reference greedy decoding without a cache (one item at a time), for testing."""
        res = []
        for b, p in enumerate(prefix):
            seq = p.to(notes.device)[None]
            n, m = notes[b:b + 1], note_mask[b:b + 1]
            gen = []
            for _ in range(min(max_new, self.cfg.max_tokens - len(p))):
                nxt = int(self(n, m, seq)[0, -1].argmax())
                if nxt == eos_id:
                    break
                gen.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], 1)
            res.append(gen)
        return res


def token_loss(logits, target, loss_mask, weights=None, label_smoothing: float = 0.0):
    """Mean cross-entropy over predicted positions (loss_mask[:, 1:]), optionally class-weighted."""
    lg = logits.float().reshape(-1, logits.shape[-1])
    y = target[:, 1:].reshape(-1)
    m = loss_mask[:, 1:].reshape(-1)
    ce = F.cross_entropy(lg, y, reduction="none", label_smoothing=label_smoothing)
    if weights is not None:
        ce = ce * weights[y]
    return (ce * m).sum() / m.sum().clamp(min=1)
