#!/bin/bash
# Round 7, extra job (on the login node from the repo root, after git pull; alongside or after
# pace/v2_round7.sh): round 6's best v2 system tuned, and seq2seq scored on this account, so the
# round-7 report compares them pair by pair.
#   bash pace/v2_round7_extra.sh
#
#   r7extra   seq2seq (the release's sf_base_reg_cab_c128, restored from models/ by
#             scripts/restore_runs.py) on whole pieces, as the handoff's best seq2seq system
#             (meter mode, the rendered MIDI's beats on clean MIDI: sf_base_reg_cab_c128_meter_sb);
#             round 6's best (nl_base_gn + meter_v4_gq_pm, PM2S's beats) deleting notes above
#             p 0.8 instead of 0.95 (round 4: slightly better on clean MIDI) and with the grid's beat
#             times taken from the notes (meter_v4_gq_pm_rf: snap 0.05 s + fill 0.25 beats; round 6:
#             -0.27 on the release, on clean MIDI)
#
# If r7final (pace/v2_round7.sh) is still pending, it is made to wait for this job too, so its
# report (scripts/analyze_round7.py) has these systems; else run it again when this job is done:
#   sbatch -J r7report -c 16 --mem=64G -t 4:00:00 -o slurm-r7report-%j.out --wrap="source pace/common.sh; \
#     module load anaconda3; eval \"\$(conda shell.bash hook)\"; conda activate bd311; \
#     python scripts/beyer_dai.py score --repo $HOME/scratch/beyer_dai/MIDI2ScoreTransformer --jobs 16 --systems all; \
#     conda activate dl-proj; python scripts/analyze_round7.py"
set -euo pipefail
GPUS=${GPUS:-cuda13}
SNAP=${SNAP:-0.05}
FILL=${FILL:-0.25}
for r in meter_v4_gq_pm nl_base_gn; do
  [ -f "outputs/runs/$r/best.pt" ] || { echo "outputs/runs/$r/best.pt missing: run pace/v2_gangqinpu.sh first"; exit 1; }
done
[ -f models/seq2seq/model.pt ] || { echo "models/seq2seq/model.pt missing: python scripts/download_models.py"; exit 1; }
JOBS="bash pace/v2_jobs.sh"
specs="nl_base_gn+meter_v4_gq_pm:tidy,d80,pm,rpm nl_base_gn+meter_v4_gq_pm_rf:tidy,d95,pm,rpm sf_base_reg_cab_c128:meter:sb"
id=$(sbatch --parsable -J r7extra -o "slurm-r7extra-%j.out" --gres=gpu:1 "--constraint=$GPUS" -c 8 --mem=48G -t 8:00:00 \
     --wrap="source pace/common.sh; python scripts/restore_runs.py --only sf_base_reg_cab_c128; \
$JOBS refine meter_v4_gq_pm meter_v4_gq_pm_rf $SNAP $FILL; $JOBS systems $specs")
echo "seq2seq + round 6's best at d80 and with beat times from the notes ($specs): job $id"
final=$(squeue -h -u "${USER:-$(whoami)}" -n r7final -t PENDING -o %i | head -1)
if [ -n "$final" ]; then
  # its dependencies still waiting ("afterany:<id>(unfulfilled),afterany:<id>(unfulfilled)"), + this job
  old=$(scontrol show job "$final" | grep -o 'Dependency=[^ ]*' | cut -d= -f2 | tr ',' '\n' \
        | grep 'unfulfilled' | sed 's/(unfulfilled)//' | paste -sd, - || true)
  new="${old:+$old,}afterany:$id"
  scontrol update JobId="$final" Dependency="$new"
  echo "r7final ($final) now also waits for $id: $(scontrol show job "$final" | sed -n 's/.*Dependency=\([^ ]*\).*/\1/p')"
else
  echo "no pending r7final: when $id is done, score and report again (the r7report command at the top of this script)"
fi
