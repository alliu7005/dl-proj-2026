#!/bin/bash
# Round 7, combinations (on the login node from the repo root, after git pull; once pace/v2_round7.sh is
# submitted, while it runs or after): round 7's labeller with round 7's new meter models, the system
# both changes make together.
#   bash pace/v2_round7_combo.sh
#
#   r7combo   whole pieces (PM2S's beats, as round 6's best): nl_base_gn_pm (the labeller trained on
#             PM2S's timing errors) with meter_v4_gq_pm_s1 (round 6's best meter model, second seed)
#             and with meter_v4_gq_pm_g30 (Gangqinpu 0.3 of every batch). pace/v2_round7.sh scores
#             nl_base_gn_pm only with meter_v4_gq_pm and meter_v3_tp.
#
# It waits for the jobs training those models (r7meters2, r7nl) if they are still queued or running,
# and a pending r7final is made to wait for it (its report then has the combinations; else score and
# report again: the r7report command at the top of pace/v2_round7_extra.sh).
set -euo pipefail
GPUS=${GPUS:-cuda13}
ME=${USER:-$(whoami)}
specs="nl_base_gn_pm+meter_v4_gq_pm_s1:tidy,d95,pm,rpm nl_base_gn_pm+meter_v4_gq_pm_g30:tidy,d95,pm,rpm"
ids=$(squeue -h -u "$ME" -n r7meters2,r7nl -o %i | paste -sd: - || true)
dep=()
[ -n "$ids" ] && dep=("--dependency=afterany:$ids")
id=$(sbatch --parsable -J r7combo -o "slurm-r7combo-%j.out" --gres=gpu:1 "--constraint=$GPUS" -c 8 --mem=48G \
     -t 8:00:00 ${dep[@]+"${dep[@]}"} --wrap="source pace/common.sh; bash pace/v2_jobs.sh systems $specs")
echo "round 7's labeller with round 7's meter models ($specs): job $id${ids:+ (waits for ${ids//:/ })}"
final=$(squeue -h -u "$ME" -n r7final -t PENDING -o %i | head -1)
if [ -n "$final" ]; then
  # its dependencies still waiting ("afterany:<id>(unfulfilled),..."), + this job
  old=$(scontrol show job "$final" | grep -o 'Dependency=[^ ]*' | cut -d= -f2 | tr ',' '\n' \
        | grep 'unfulfilled' | sed 's/(unfulfilled)//' | paste -sd, - || true)
  scontrol update JobId="$final" Dependency="${old:+$old,}afterany:$id"
  echo "r7final ($final) now also waits for $id: $(scontrol show job "$final" | sed -n 's/.*Dependency=\([^ ]*\).*/\1/p')"
else
  echo "no pending r7final: when $id is done, score and report again (the r7report command in pace/v2_round7_extra.sh)"
fi
