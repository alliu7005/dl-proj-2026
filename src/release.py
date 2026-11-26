"""The released models: what is in models/ (the layout of the Hugging Face model repo), and where
the demo and scripts/audio_to_score.py look for a model: models/ first (scripts/download_models.py),
else the training run it came from (outputs/runs/, what a training machine has).

    models/
      README.md                  the model card (docs/model_card.md)
      manifest.json              what each file is, the run it came from, its sha256
      v2_labeller/  model.pt  config.json  vocab.json
      v2_meter/     model.pt  config.json  decode.json    (the tuned decoder, tempo prior included)
      seq2seq/      model.pt  config.json  vocab.json  feature_stats.json

scripts/export_models.py makes it from the training runs; scripts/download_models.py fetches it.
scripts/publish_model.py puts one finished run in its slot on the Hugging Face repo when it beats the
model there (pace/v2_jobs.sh calls it after every training run), and keeps every run under runs/;
the demo pulls the repo's current default when it starts and every hour (download_models.update).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path("models")

# The Hugging Face model repo the release is uploaded to (scripts/download_models.py --repo overrides it).
HF_REPO = "dat-dev/scoreformer"

# name -> its file in models/, and the training run's checkpoint used when models/ does not have it.
# The release's sources (export_models.py defaults) are the round-4 winners (docs/experiments.md,
# "Fourth round: results"), with the meter model trained with Gangqinpu since 2026-11-26 (sixth
# round: Gangqinpu's real recordings' time signatures right 0.79 of the time against 0.54, classical
# meter scores equal); the local fallbacks are what a training machine has.
RELEASE = {
    "v2_labeller": {"file": "v2_labeller/model.pt", "run": "outputs/runs/nl_base_gn/best.pt",
                    "what": "v2 note labeller (nl_base_gn)"},
    "v2_meter": {"file": "v2_meter/model.pt", "run": "outputs/runs/meter_v4_gq/best.pt",
                 "what": "v2 meter model (meter_v4_gq: trained with Gangqinpu, its tempo-prior decoder)"},
    "seq2seq": {"file": "seq2seq/model.pt", "run": "outputs/runs/sf_base_reg_cab/best.pt",
                "what": "seq2seq score model (sf_base_reg_cab_c128)"},
}

# The settings the release is meant to run with (the demo's and audio_to_score.py's defaults): keep a
# note unless the labeller is at least this sure it is not in the score, trim written lengths
# that run just past the next onset (src/pipeline.py tidy_lengths), write glissandos as glissandos.
DELETE_ABOVE = 0.95
TIDY = True
# glissandos written as a "gliss." line between their first and last note (src/glissando.py)
GLISSANDO = True


def default_path(name: str) -> str:
    """models/<file> if it is there, else the training run's checkpoint (paths relative to the repo)."""
    rel = MODELS / RELEASE[name]["file"]
    return str(rel if (ROOT / rel).exists() else Path(RELEASE[name]["run"]))
