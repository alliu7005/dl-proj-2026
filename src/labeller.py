"""The note labeller (v2): one pass over the performed notes, one set of score labels per note.

Instead of writing the score token by token (src/scoreformer.py), the labeller reads the notes of
a window (plus context notes) together with where a metrical grid puts each of them
(src/notelabels.py grid_features), and predicts per note:
  keep     is it a score note (else a transcription error, or an extra note of a trill / tremolo)
  delta    the bar correction to the grid's bar (-1, 0, +1): notes played just before a bar line
  pos, sub its position in the bar, as the target grammar writes it (pos_k + sub_x)
  dur      its written length, ties included (src/notelabels.py render splits it at bar lines)
  staff    0 .. 3
  flags    grace, arpeggio, fermata, mordent, tremolo, trill, turn (independent yes / no)
No generation: nothing can loop, no window has to guess where in the bar it starts, and the whole
window is labelled in one parallel pass. The encoder is the Scoreformer's (time rotary attention
over onsets in seconds, binned note embeddings); the grid features enter with the other per-note
flags. ``placed_notes`` turns the predictions back into notes the renderer writes as tokens.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .notelabels import N_FLAGS, N_POS, Placed, unflag
from .scoreformer import EncoderLayer, NoteEmbedding, ScoreformerConfig, rope_angles

PRESETS = {
    "small": dict(d_model=256, n_heads=4, enc_layers=4, ff=1024),
    "base": dict(d_model=512, n_heads=8, enc_layers=8, ff=2048),
    "large": dict(d_model=768, n_heads=12, enc_layers=12, ff=3072),
}
N_STAFF = 4


@dataclass
class LabellerConfig:
    n_sub: int
    n_dur: int
    n_features: int
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 8
    ff: int = 2048
    dropout: float = 0.1
    max_notes: int = 1024
    feat_mean: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    feat_std: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    time_min_period: float = 0.02
    time_max_period: float = 60.0
    n_time_bins: int = 64
    n_vel_bins: int = 32
    context_before: int = 0
    context_after: int = 0
    n_pos: int = N_POS
    n_staff: int = N_STAFF
    n_flags: int = N_FLAGS

    def to_dict(self):
        return asdict(self)

    def heads(self) -> list[tuple[str, int]]:
        return [("keep", 1), ("delta", 3), ("pos", self.n_pos), ("sub", self.n_sub), ("dur", self.n_dur),
                ("staff", self.n_staff), ("flags", self.n_flags)]


class NoteLabeller(nn.Module):
    arch = "labeller"

    def __init__(self, c: LabellerConfig):
        super().__init__()
        self.cfg = c
        enc = ScoreformerConfig(vocab_size=1, d_model=c.d_model, n_heads=c.n_heads, enc_layers=c.enc_layers,
                                dec_layers=0, ff=c.ff, dropout=c.dropout, max_notes=c.max_notes,
                                feat_mean=c.feat_mean, feat_std=c.feat_std, time_min_period=c.time_min_period,
                                time_max_period=c.time_max_period, n_time_bins=c.n_time_bins,
                                n_vel_bins=c.n_vel_bins, n_features=c.n_features)
        self.embed = NoteEmbedding(enc)
        self.enc = nn.ModuleList(EncoderLayer(enc) for _ in range(c.enc_layers))
        self.norm = nn.LayerNorm(c.d_model)
        self.sizes = c.heads()
        self.head = nn.Linear(c.d_model, sum(n for _, n in self.sizes))
        dh = c.d_model // c.n_heads
        periods = torch.logspace(math.log10(c.time_min_period), math.log10(c.time_max_period), dh // 2)
        self.register_buffer("time_freq", 2 * math.pi / periods, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
                if name.endswith(("attn.o.weight", "w3.weight")):
                    p.data.mul_(1 / math.sqrt(2 * c.enc_layers))
            elif name.endswith(".bias"):
                nn.init.zeros_(p)
        with torch.no_grad():
            self.embed.pitch.weight[0].zero_()

    def forward(self, notes, note_mask) -> dict[str, torch.Tensor]:
        """{head: logits [B, N, size]} (keep: [B, N] logit, flags: [B, N, n_flags] logits)."""
        h, onset = self.embed(notes, note_mask)
        cos, sin = rope_angles(onset, self.time_freq)
        mask = note_mask[:, None, None, :]
        for layer in self.enc:
            h = layer(h, cos, sin, mask)
        out = self.head(self.norm(h))
        res, k = {}, 0
        for name, n in self.sizes:
            res[name] = out[..., k:k + n]
            k += n
        res["keep"] = res["keep"][..., 0]
        return res


def labeller_loss(out: dict, labels: torch.Tensor, flags: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Sum of the heads' losses over the notes each has a target for (-1 = none): binary
    cross-entropy for keep and the flags, cross-entropy for the rest. Returns (loss, per head)."""
    parts = {}
    zero = out["keep"].sum() * 0.0
    m = labels[..., 0] >= 0
    parts["keep"] = (F.binary_cross_entropy_with_logits(out["keep"][m].float(), labels[..., 0][m].float())
                     if m.any() else zero)
    for col, name in enumerate(("delta", "pos", "sub", "dur", "staff"), start=1):
        m = labels[..., col] >= 0
        parts[name] = F.cross_entropy(out[name][m].float(), labels[..., col][m]) if m.any() else zero
    m = flags >= 0
    parts["flags"] = (F.binary_cross_entropy_with_logits(out["flags"][m].float(), flags[m].float())
                      if m.any() else zero)
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}


def head_counts(out: dict, labels: torch.Tensor) -> dict:
    """Correct / labelled counts per head (accuracy = sum of correct / sum of n, by the caller)."""
    c = {}
    m = labels[..., 0] >= 0
    c["keep_ok"] = int(((out["keep"] > 0).long() == labels[..., 0])[m].sum())
    c["keep_n"] = int(m.sum())
    for col, name in enumerate(("delta", "pos", "sub", "dur", "staff"), start=1):
        m = labels[..., col] >= 0
        c[f"{name}_ok"] = int((out[name].argmax(-1) == labels[..., col])[m].sum())
        c[f"{name}_n"] = int(m.sum())
    return c


def keep_mask(keep_logits: torch.Tensor, delete_above: float = 0.5) -> torch.Tensor:
    """The notes kept: a note is deleted only when the keep head's probability of "delete" is above
    ``delete_above`` (0.5: the head's own decision, as trained and evaluated; 1 or more: none).
    ``delete_above`` trades notes wrongly left out against transcription errors kept: the "delete"
    label is "the alignment paired no score note with it", which also covers notes really played
    (on clean MIDI, 34% of what the head deletes at 0.5 had a score note; at 0.8, 16%; at 0.95, 4%)."""
    if delete_above >= 1.0:
        return torch.ones_like(keep_logits, dtype=torch.bool)
    t = float(np.log(delete_above / (1.0 - delete_above)))  # p(delete) > d  <=>  logit(keep) < -logit(d)
    return keep_logits > -t


def placed_notes(out: dict, b: int, pitches, bar_est, own: np.ndarray, space, staves: int = 0,
                 delete_above: float = 0.5) -> list[Placed]:
    """Item ``b`` of a batch's predictions -> the notes to render: every own note (window or
    carried, ``own`` mask over the item's notes; context notes belong to other windows) the keep
    head keeps, at its predicted bar (grid bar + correction, never before bar 0), position,
    length, staff and flags. ``pitches`` / ``bar_est``: the item's notes (performed pitches).
    ``staves``: only the first that many staves (each note on its likeliest of them; 0 = all).
    ``delete_above``: see keep_mask."""
    n = min(len(own), out["keep"].shape[1])  # the batch pads every item to its longest one
    keep = keep_mask(out["keep"][b, :n], delete_above).cpu().numpy()
    pick = {k: out[k][b, :n].argmax(-1).cpu().numpy() for k in ("delta", "pos", "sub", "dur")}
    staff_logits = out["staff"][b, :n]
    if staves:
        staff_logits = staff_logits[:, :staves]
    pick["staff"] = staff_logits.argmax(-1).cpu().numpy()
    flags = (out["flags"][b, :n] > 0).cpu().numpy()
    res = []
    for i in np.nonzero(np.asarray(own[:n], bool) & keep)[0]:
        grace, arp, orns = unflag(int(sum(int(v) << k for k, v in enumerate(flags[i]))))
        bar = max(0, int(bar_est[i]) + int(pick["delta"][i]) - 1)
        res.append(Placed(bar, space.pos_value(pick["pos"][i], pick["sub"][i]), int(round(float(pitches[i]))),
                          space.dur_value(pick["dur"][i]), int(pick["staff"][i]), grace, arp, orns))
    return res


def build_labeller(cfg: dict) -> NoteLabeller:
    return NoteLabeller(LabellerConfig(**cfg))


def load_labeller(path, device) -> NoteLabeller:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_labeller(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def predict_placed(model, items: list[dict], coll, device, amp: bool, space, batch_size: int = 32,
                   carried: bool = True, staves: int = 0, delete_above: float = 0.5) -> list[list[Placed]]:
    """Label dataset items (ScoreSegmentDataset(notelabels=True), or src/pipeline.py V2Pipeline's
    windows): the notes each keeps, placed in its window's bars. ``carried``: also the notes carried
    over from the window before (a window's target writes their continuation; a whole piece
    places each note once, in its own window). ``staves``, ``delete_above``: see placed_notes."""
    from .dataset import ROLE_CARRIED, ROLE_WINDOW
    model.eval()
    res = [None] * len(items)
    order = sorted(range(len(items)), key=lambda k: len(items[k]["notes"]))
    for lo in range(0, len(order), batch_size):
        idx = order[lo:lo + batch_size]
        b = coll([dict(items[i]) for i in idx])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["notes"].to(device), b["note_mask"].to(device))
        out = {k: v.float() for k, v in out.items()}
        for r, i in enumerate(idx):
            it = items[i]
            own = (it["roles"] <= (ROLE_CARRIED if carried else ROLE_WINDOW)).numpy()
            res[i] = placed_notes(out, r, it["notes"][:, 0].numpy(), it["bar_est"].numpy(), own, space, staves,
                                  delete_above)
    return res


def predict_windows(model, items: list[dict], windows: list, coll, device, amp: bool, space,
                    batch_size: int = 32) -> list[list[str]]:
    """Label dataset items (ScoreSegmentDataset(notelabels=True)) and render each as tokens in the
    frame ``windows[k]`` (src/notelabels.py Window: its bars' time / key signatures; Phase 1 gives
    the target's own). Returns one token list per item, in order. The window's own notes only:
    a note carried over from the window before has no label to learn (its continuation is written
    by the window it was struck in, as V2Pipeline renders a piece, and as the label ceiling counts
    it), so its untrained outputs are not rendered."""
    from .notelabels import render
    placed = predict_placed(model, items, coll, device, amp, space, batch_size, carried=False)
    return [render(w, pl) for w, pl in zip(windows, placed)]
