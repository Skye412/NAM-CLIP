#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

SRC="/home/skye/data/Skye/Tf_MoeClip/src"
OUT="/home/skye/data/Skye/Tf_MoeClip/outputs/retrain_traceable_20261105_s111/external_sota"
mkdir -p "$OUT"
LOG="$OUT/master_ext.log"
echo $$ > "$OUT/master_ext.pid"

echo "=== [$(date)] Master PID=$$ Starting Step 1 ===" >> "$LOG"

for item in "adaptclip:adaptclip" "adaptclip:anomalyclip" "Bayes_PFL:bayespfl" "wpformer:wpformer" "MixerCSeg:mixercseg" "dinomaly:dinomaly"; do
    env_name="${item%%:*}"
    method="${item##*:}"
    if [ ! -f "$OUT/metrics_comp_${method}_Bridge512.json" ]; then
        echo "--- [$(date)] Step 1: $method (env=$env_name) ---" >> "$LOG"
        /home/skye/miniconda3/envs/$env_name/bin/python -u "$SRC/eval_ext_step1.py" "$method" >> "$LOG" 2>&1
    fi
done

echo "=== [$(date)] Step 1 All 6 Done! Starting Step 2 (Train & Eval on Official S2DS) ===" >> "$LOG"

for item in "adaptclip:adaptclip" "adaptclip:anomalyclip" "Bayes_PFL:bayespfl" "wpformer:wpformer" "MixerCSeg:mixercseg" "dinomaly:dinomaly"; do
    env_name="${item%%:*}"
    method="${item##*:}"
    if [ ! -f "$OUT/metrics_comp_${method}_S2DS.json" ]; then
        echo "--- [$(date)] Step 2: $method (env=$env_name) ---" >> "$LOG"
        /home/skye/miniconda3/envs/$env_name/bin/python -u "$SRC/train_ext_step2.py" "$method" >> "$LOG" 2>&1
    fi
done

echo "=== [$(date)] ALL 12 EXTERNAL SOTA EVALUATIONS & TRAININGS COMPLETED! ===" >> "$LOG"
