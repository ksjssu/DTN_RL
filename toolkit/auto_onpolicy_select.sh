#!/usr/bin/env bash
set -euo pipefail

# Auto pipeline: on-policy TBPTT PPO streaming updates for training,
# episode-end final delivery rate for selection/evaluation.
#
# What it does
# 1) Launch training server (EPISODE_UPDATE=0)
# 2) For each exploration seed: restore baseline -> run 1 episode -> save model -> parse final delivery rate
# 3) Pick top-K candidates by final delivery rate
# 4) For each candidate: launch eval-only server and evaluate over multiple seeds, compute median/LCB
# 5) Print summary and best candidate by median (ties by LCB)
#
# Requirements
# - ./compile.sh has been run
# - one.sh callable; scenario file is valid
# - curl available

usage() {
  cat <<EOF
Usage: $0 \\
  --scenario <path> [--port 5010] [--eval-port 5510] \\
  [--explore-seeds "1001,1002,1003"] [--explore-n 20] \\
  [--explore-episodes 1] [--score last|best] \\
  [--topk 5] [--eval-seeds "2001,2002,2003"] [--outdir reports/auto]

Notes:
- If --explore-seeds is given, it overrides --explore-n.
- The script creates temp scenario copies with RLBridgeReport.url, RLStateReport.reportUrl,
  and MovementModel.rngSeed adjusted per run.
- Training server runs in streaming mode (EPISODE_UPDATE=0).
- Evaluation server runs with EVAL_ONLY=1 per candidate model.
EOF
}

# Defaults
PORT=5010
EVAL_PORT=5510
SCEN=""
OUTDIR="reports/auto"
EXP_SEEDS=""
EXP_N=20
EXP_EPISODES=1
SCORE_MODE="last"  # last|best
TOPK=5
EVAL_SEEDS="2001,2002,2003"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCEN="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --eval-port) EVAL_PORT="$2"; shift 2;;
    --explore-seeds) EXP_SEEDS="$2"; shift 2;;
    --explore-n) EXP_N="$2"; shift 2;;
    --explore-episodes) EXP_EPISODES="$2"; shift 2;;
    --score) SCORE_MODE="$2"; shift 2;;
    --topk) TOPK="$2"; shift 2;;
    --eval-seeds) EVAL_SEEDS="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$SCEN" ]]; then echo "--scenario required"; usage; exit 1; fi
if [[ ! -f "$SCEN" ]]; then echo "Scenario not found: $SCEN"; exit 1; fi

mkdir -p "$OUTDIR"
TMPDIR=".tmp/auto_onpolicy"
mkdir -p "$TMPDIR"

# Helpers
log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_server() {
  local port="$1"; local tries=50
  for i in $(seq 1 $tries); do
    if curl -s -o /dev/null -X POST "http://127.0.0.1:${port}/restore_baseline" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

make_scen_copy() {
  local in="$1"; local out="$2"; local port="$3"; local seed="$4"
  cp -f "$in" "$out"
  # Ensure RLBridgeReport.url points to our server
  if grep -q '^RLBridgeReport\.url' "$out"; then
    sed -i -E "s#^RLBridgeReport\.url\s*=.*#RLBridgeReport.url = http://127.0.0.1:${port}/infer_and_update#g" "$out"
  else
    printf "\nRLBridgeReport.url = http://127.0.0.1:%s/infer_and_update\n" "$port" >> "$out"
  fi
  # Ensure RLStateReport.reportUrl posts to /episode_end (for clean bookkeeping)
  if grep -q '^RLStateReport\.reportUrl' "$out"; then
    sed -i -E "s#^RLStateReport\.reportUrl\s*=.*#RLStateReport.reportUrl = http://127.0.0.1:${port}/episode_end#g" "$out"
  else
    printf "\nRLStateReport.reportUrl = http://127.0.0.1:%s/episode_end\n" "$port" >> "$out"
  fi
  # Set MovementModel.rngSeed (add if missing)
  if grep -q '^MovementModel\.rngSeed' "$out"; then
    sed -i -E "s#^MovementModel\.rngSeed\s*=.*#MovementModel.rngSeed = ${seed}#g" "$out"
  else
    printf "\nMovementModel.rngSeed = %s\n" "$seed" >> "$out"
  fi
}

parse_episode_reward() {
  # Return: avg_delivery_rate,delivered,created from latest *_EpisodeReward.txt
  # Search under reports/ and reports_* recursively, pick newest file.
  local latest
  latest=$(find reports reports_* -type f -name "*_EpisodeReward.txt" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -n 1 || true)
  if [[ -z "$latest" ]]; then echo ",,"; return 0; fi
  local line
  line=$(grep -h "^average_delivery_rate" "$latest" | tail -n 1 || true)
  if [[ -z "$line" ]]; then echo ",,"; return 0; fi
  # Format: average_delivery_rate <val> delivered <d> created <c>
  local rate del created
  rate=$(echo "$line" | awk '{for(i=1;i<=NF;i++){if($i=="average_delivery_rate"){print $(i+1);break}}}')
  del=$(echo "$line" | awk '{for(i=1;i<=NF;i++){if($i=="delivered"){print $(i+1);break}}}')
  created=$(echo "$line" | awk '{for(i=1;i<=NF;i++){if($i=="created"){print $(i+1);break}}}')
  echo "${rate},${del},${created}"
}

compute_stats() {
  # stdin: one value per line
  # prints: mean,sd,median,lcb
  awk '
    {x[NR]=$1; s+=$1; ss+=$1*$1}
    END{
      n=NR; if(n==0){print "0,0,0,0"; exit}
      mean=s/n; var=(n>1? (ss - s*s/n)/(n-1):0); sd=sqrt(var)
      # median
      asort(x)
      if (n%2==1) med=x[(n+1)/2]; else med=(x[n/2]+x[n/2+1])/2
      lcb=mean - 1.96*(sd/sqrt(n))
      printf("%f,%f,%f,%f\n", mean, sd, med, lcb)
    }
  '
}

# Build exploration seed list
IFS="," read -r -a EXP_SEEDS_ARR <<< "${EXP_SEEDS}"
if [[ -z "${EXP_SEEDS}" ]]; then
  EXP_SEEDS_ARR=()
  for i in $(seq 1 "$EXP_N"); do EXP_SEEDS_ARR+=($((1000+i))); done
fi

TRAIN_LOG="$OUTDIR/train_server_${PORT}.log"
EVAL_LOG_BASE="$OUTDIR/eval_server"

log "Launching training server on port ${PORT} (streaming updates)"
(
  EPISODE_UPDATE=0 DRL_PORT="$PORT" EVAL_ONLY=0 \
  MIN_SEQUENCES_TO_TRAIN=${MIN_SEQUENCES_TO_TRAIN:-128} \
  PPO_MINIBATCH=${PPO_MINIBATCH:-1024} PPO_EPOCHS=${PPO_EPOCHS:-1} \
  MAX_SEQS_PER_UPDATE=${MAX_SEQS_PER_UPDATE:-2048} \
  python toolkit/drl_server_rmappo.py
) >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
trap 'kill $TRAIN_PID >/dev/null 2>&1 || true' EXIT

if ! wait_server "$PORT"; then
  echo "Failed to bring up training server on port $PORT" >&2; exit 1
fi
log "Training server ready (pid=$TRAIN_PID)"

EXP_CSV="$OUTDIR/exp_results.csv"
echo "seed,model,avg_delivery_rate,delivered,created" > "$EXP_CSV"

for SEED in "${EXP_SEEDS_ARR[@]}"; do
  RUN_SCEN="$TMPDIR/exp_seed_${SEED}.txt"
  make_scen_copy "$SCEN" "$RUN_SCEN" "$PORT" "$SEED"
  log "[EXP] seed=${SEED} -> ${EXP_EPISODES} episode(s) training run"
  # Reset to baseline
  curl -s -X POST "http://127.0.0.1:${PORT}/restore_baseline" >/dev/null || true
  # Run N episodes and score
  LAST_AVG=""; LAST_DEL=""; LAST_CRE=""; BEST_AVG=""
  : >"$OUTDIR/sim_seed_${SEED}.log"
  for EPI in $(seq 1 "$EXP_EPISODES"); do
    log "  - episode ${EPI}/${EXP_EPISODES}"
    ./one.sh -b 3 "$RUN_SCEN" >>"$OUTDIR/sim_seed_${SEED}.log" 2>&1 || true
    REWARD_LINE=$(parse_episode_reward)
    AVG=$(echo "$REWARD_LINE" | cut -d, -f1)
    DEL=$(echo "$REWARD_LINE" | cut -d, -f2)
    CRE=$(echo "$REWARD_LINE" | cut -d, -f3)
    LAST_AVG="$AVG"; LAST_DEL="$DEL"; LAST_CRE="$CRE"
    if [[ -z "$BEST_AVG" ]]; then BEST_AVG="$AVG"; fi
    if [[ -n "$AVG" && -n "$BEST_AVG" ]]; then
      # Use awk inside IF so set -e won't abort on non-zero status
      if awk -v a="$AVG" -v b="$BEST_AVG" 'BEGIN{ exit (a+0 > b+0)?0:1 }'; then
        BEST_AVG="$AVG"
      fi
    fi
  done
  if [[ "$SCORE_MODE" == "best" ]]; then USE_AVG="$BEST_AVG"; else USE_AVG="$LAST_AVG"; fi
  # Save model with a unique name
  MODEL_PATH="models_rmappo/auto_exp_seed_${SEED}.pt"
  curl -s -X POST -H 'Content-Type: application/json' \
       -d "{\"path\":\"${MODEL_PATH}\"}" \
       "http://127.0.0.1:${PORT}/save_model" >/dev/null || true
  echo "${SEED},${MODEL_PATH},${USE_AVG},${LAST_DEL},${LAST_CRE}" >> "$EXP_CSV"
done

log "Exploration finished. Picking top-${TOPK} by avg_delivery_rate"
BEST_CSV="$OUTDIR/exp_topk.csv"
tail -n +2 "$EXP_CSV" | sort -t, -k3,3nr | head -n "$TOPK" > "$BEST_CSV"

read -r -a EVAL_SEEDS_ARR <<< "${EVAL_SEEDS}"

BEST_SUMMARY="$OUTDIR/eval_summary.csv"
echo "model,mean,sd,median,lcb,n" > "$BEST_SUMMARY"

rank=0
while IFS=, read -r seed model avg del cre; do
  rank=$((rank+1))
  log "[EVAL] Candidate #${rank}: $model"
  # Launch eval server for this model
  ELOG="$EVAL_LOG_BASE_${rank}_${EVAL_PORT}.log"
  (
    DRL_PORT="$EVAL_PORT" EVAL_ONLY=1 MODEL_PATH="$model" python toolkit/drl_server_rmappo.py
  ) >"$ELOG" 2>&1 &
  EPID=$!
  trap 'kill $TRAIN_PID >/dev/null 2>&1 || true; kill $EPID >/dev/null 2>&1 || true' EXIT
  if ! wait_server "$EVAL_PORT"; then
    echo "Eval server failed on port $EVAL_PORT" >&2; kill $EPID || true; continue
  fi

  # Run eval seeds and collect final rates
  TMPVALS="$TMPDIR/eval_vals_${rank}.txt"
  : > "$TMPVALS"
  for ESEED in "${EVAL_SEEDS_ARR[@]}"; do
    RUN_SCEN="$TMPDIR/eval_candidate${rank}_seed_${ESEED}.txt"
    make_scen_copy "$SCEN" "$RUN_SCEN" "$EVAL_PORT" "$ESEED"
    log "  - eval seed=${ESEED}"
    ./one.sh -b 3 "$RUN_SCEN" >"$OUTDIR/eval_candidate${rank}_seed_${ESEED}.log" 2>&1 || true
    REWARD_LINE=$(parse_episode_reward)
    AVG=$(echo "$REWARD_LINE" | cut -d, -f1)
    if [[ -n "$AVG" ]]; then echo "$AVG" >> "$TMPVALS"; fi
  done
  kill $EPID >/dev/null 2>&1 || true

  if [[ -s "$TMPVALS" ]]; then
    STATS=$(compute_stats < "$TMPVALS")
    MEAN=$(echo "$STATS" | cut -d, -f1)
    SD=$(echo "$STATS"   | cut -d, -f2)
    MED=$(echo "$STATS"  | cut -d, -f3)
    LCB=$(echo "$STATS"  | cut -d, -f4)
    N=$(wc -l < "$TMPVALS")
    echo "${model},${MEAN},${SD},${MED},${LCB},${N}" >> "$BEST_SUMMARY"
  else
    echo "${model},0,0,0,0,0" >> "$BEST_SUMMARY"
  fi
done < "$BEST_CSV"

log "Ranking candidates by median (desc), tie-break by LCB"
BEST_PICK="$OUTDIR/final_pick.csv"
tail -n +2 "$BEST_SUMMARY" | sort -t, -k4,4nr -k5,5nr | head -n 1 > "$BEST_PICK"
log "Done. Files:"
echo " - Exploration results: $EXP_CSV"
echo " - Top-K candidates:    $BEST_CSV"
echo " - Eval summary:        $BEST_SUMMARY"
echo " - Final pick:          $BEST_PICK"

log "Best model (median-first):"
cat "$BEST_PICK"
