#!/bin/bash
# Rebuild data/processed with the 2026-11-22 fixes to how scores become training data (README,
# "Repo sweep, 2026-11-22"): ties in chords, pickups, one piece in train and val. On the login node
# from the repo root, after git pull:
#   bash pace/rebuild.sh                  # one CPU job (up to 8 h); prints its id
#   AFTER=<that id> bash pace/v2.sh       # the third round on the rebuilt data, once it is done
#
# The current data is kept as $OLD (data/processed_pre1122): nothing is deleted. The rebuild keeps
# its vocabulary (the same token ids: every checkpoint reads the new data) and its feature scaling
# (the performed notes are the same). What depends on the scores is built again from data/raw: the
# windows and targets (build_dataset.py), the real-transcription splits (build_transcribed.py: the
# Transkun / Kong transcriptions themselves are reused), the metrical labels, the annotated beats,
# the note labels and the meter model's data. Beat This!'s tracked beats are copied over (the same
# recordings). Then the data tests; the job fails if any step does.
set -euo pipefail
OLD=${OLD:-data/processed_pre1122}
if [ -e "$OLD" ]; then
  echo "$OLD exists already (rebuilt before?): move it away or set OLD=<another folder>"
  exit 1
fi
[ -f data/processed/vocab.json ] || { echo "no data/processed/vocab.json: nothing to rebuild from"; exit 1; }
id=$(sbatch --parsable -J rebuild -c 8 --mem=64G -t 8:00:00 -o slurm-rebuild-%j.out --wrap="source pace/common.sh; \
set -euo pipefail; \
mv data/processed $OLD; \
python scripts/build_dataset.py --vocab $OLD/vocab.json; \
cp $OLD/feature_stats.json data/processed/feature_stats.json; \
python scripts/build_transcribed.py --asap-split test; \
python scripts/build_transcribed.py --asap-split train --maestro-splits test,validation; \
python scripts/build_transcribed.py --asap-split val --maestro-splits test,validation; \
if [ -d $OLD/test_transcribed_kong ]; then python scripts/build_transcribed.py --asap-split test --transcribed data/transcribed; fi; \
for s in train_transcribed val_transcribed test_transcribed test_transcribed_kong; do \
  if [ -d $OLD/\$s ] && [ -d data/processed/\$s ]; then cp $OLD/\$s/beats_tracked.* data/processed/\$s/ 2>/dev/null || true; fi; \
done; \
python scripts/build_metrical.py --workers 8; \
python scripts/build_beats.py --workers 8 --check; \
python scripts/build_notelabels.py --workers 8; \
python scripts/build_meter_data.py --workers 8 --check; \
python -m tests.test_integrity; python -m tests.test_preprocessing; \
du -sh data/processed $OLD")
echo "rebuild: job $id (log slurm-rebuild-$id.out); the data in use is kept as $OLD"
echo "then: AFTER=$id bash pace/v2.sh"
