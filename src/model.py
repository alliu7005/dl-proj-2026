"""Encoder-decoder transformer: performed notes -> score tokens.

Encoder input: the Dataset's per-note features [B, N, 8] (see ``dataset.note_features``):
    0 pitch (MIDI, raw)          -> embedding (88 keys + padding)
    1 onset (standardized)       -> Fourier features (time position) + linear
    2-7 ioi, log duration, velocity (standardized), pedal at onset / release, carried-over (0/1)
                                 -> linear
plus a sinusoidal encoding of the note's index in the window (notes are sorted by onset).

Decoder: token embedding + learned positions, causal self-attention, cross-attention to the notes.
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

    def forward(self, x, mask):
        """x [B,N,8], mask [B,N] True = real note -> memory [B,N,D]."""
        n = x.shape[1]
        pitch = torch.where(mask, (x[..., 0] - 20).clamp(1, 89), torch.zeros_like(x[..., 0])).long()
        ang = x[..., 1:2] * self.freqs
        h = (self.pitch(pitch) + self.cont(x[..., 1:]) + self.fourier(torch.cat([ang.sin(), ang.cos()], -1))
             + self.pos[:n])
        h = self.drop(self.norm(h))
        return self.layers(h, src_key_padding_mask=~mask)


class Score2Seq(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.encoder = NoteEncoder(c)
        self.tok = nn.Embedding(c.vocab_size, c.d_model, padding_idx=0)
        self.tok_pos = nn.Embedding(c.max_tokens, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        layer = nn.TransformerDecoderLayer(c.d_model, c.n_heads, c.ff, c.dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, c.dec_layers)
        self.out_norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        # small init: nn.Embedding's default N(0, 1) makes the tied output logits huge
        for emb in (self.tok, self.tok_pos, self.encoder.pitch):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.tok.weight[0].zero_()
            self.encoder.pitch.weight[0].zero_()

    def decode(self, tokens, memory, mem_mask):
        t = tokens.shape[1]
        h = self.drop(self.tok(tokens) + self.tok_pos(torch.arange(t, device=tokens.device)))
        causal = torch.ones(t, t, dtype=torch.bool, device=tokens.device).triu(1)  # True = masked
        h = self.decoder(h, memory, tgt_mask=causal, tgt_is_causal=True,
                         tgt_key_padding_mask=tokens == 0, memory_key_padding_mask=~mem_mask)
        return self.head(self.out_norm(h))

    def forward(self, notes, note_mask, dec_in):
        return self.decode(dec_in, self.encoder(notes, note_mask), note_mask)

    @torch.no_grad()
    def generate(self, notes, note_mask, prefix: list[torch.Tensor], eos_id: int, max_new: int = 1200):
        """Greedy decoding. ``prefix[b]`` = <bos> [prompt <sep>] for item b. Returns, per item,
        the generated token ids after the prefix (without <eos>)."""
        B, dev = len(prefix), notes.device
        memory = self.encoder(notes, note_mask)
        lens = torch.tensor([len(p) for p in prefix], device=dev)
        T = min(int(lens.max()) + max_new, self.cfg.max_tokens)
        seq = torch.zeros(B, T, dtype=torch.long, device=dev)
        for b, p in enumerate(prefix):
            seq[b, :len(p)] = p.to(dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        cur = lens.clone()
        while not done.all() and int(cur.max()) < T:
            L = int(cur.max())
            logits = self.decode(seq[:, :L], memory, note_mask)
            nxt = logits[torch.arange(B, device=dev), cur - 1].argmax(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            write = ~done & (cur < T)
            seq[write, cur[write]] = nxt[write]
            done |= nxt == eos_id
            cur = torch.where(write, cur + 1, cur)
        out = []
        for b in range(B):
            g = seq[b, lens[b]:cur[b]].tolist()
            out.append(g[:g.index(eos_id)] if eos_id in g else g)
        return out


def token_loss(logits, target, loss_mask, weights=None, label_smoothing: float = 0.0):
    """Mean cross-entropy over predicted positions (loss_mask[:, 1:]), optionally class-weighted."""
    lg = logits.float().reshape(-1, logits.shape[-1])
    y = target[:, 1:].reshape(-1)
    m = loss_mask[:, 1:].reshape(-1)
    ce = F.cross_entropy(lg, y, reduction="none", label_smoothing=label_smoothing)
    if weights is not None:
        ce = ce * weights[y]
    return (ce * m).sum() / m.sum().clamp(min=1)
