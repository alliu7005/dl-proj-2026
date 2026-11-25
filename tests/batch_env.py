"""The environment the tests of the pace/*.sh batch scripts run those scripts in.

The batch scripts read their settings from environment variables (METERS, NLS, SYSTEMS, ...), and
pace/v2.sh exports some of them to its jobs, its test job included: a test that ran a batch script
in the test job's own environment would get the batch's settings instead of the defaults it checks
(pace/meter.sh then saw METERS=meter_v3 and failed). Every batch test starts from this instead.
"""
from __future__ import annotations

import os

BATCH_VARS = (
    # pace/v2.sh, pace/v2_jobs.sh
    "METERS", "NEW_NLS", "NLS", "NLS_EXTRA", "GRIDS_OLD", "GRIDS_REEVAL", "METER_GROUPS", "SYSTEMS", "SEQ_SB",
    "SEQ_RERUN", "SKIP_SYSTEM", "AFTER", "MAX_HOURS", "BD",
    # pace/meter.sh, pace/labeller.sh, pace/rebuild.sh
    "LABELLERS", "LARGE_ON", "SKIP_PREP", "OLD",
    # pace/reevaluate.sh, pace/beats.sh, pace/tsearch.sh, pace/system_eval.sh, pace/train*.sbatch
    "RUNS", "ONLY", "ARGS", "FORCE", "PER_CONDITION", "TIME", "STAGE", "NAME", "GPUS",
    # pace/v2_round4.sh, pace/v2_release.sh, pace/v2_level.sh
    "NEW_LABS", "TEMPO_FROM", "TEMPO_RUN", "SYS_A", "SYS_B", "SYS_C", "ABLATION", "SYS", "EXPORT", "LABELLER",
    "LEVELEXP",
    # pace/v2_gangqinpu.sh
    "BASE", "SNAP", "FILL", "GQ", "PM_PAIR", "AUDIO_TRAIN", "PM2S_RUNS",
    # pace/v2_pm2s.sh, and which PM2S weights the jobs run (src/pm2s_beats.py weights_tag)
    "PM_RUN", "PM_TAG", "METER", "PM2S_WEIGHTS", "PM2S_TAG",
    # the tests' own stand-ins
    "SUBMITS", "CALLS", "DONE",
)


def batch_env(**extra) -> dict:
    """os.environ without any batch setting, plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if k not in BATCH_VARS}
    env.update(extra)
    return env
