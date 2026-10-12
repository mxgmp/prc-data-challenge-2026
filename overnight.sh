#!/usr/bin/env bash
# model_v8.py: learning rate 0.025, 127 leaves, AOBT_3 runs (NOTEBOOK entry 26). Run: bash overnight.sh
BASE=logs/pred_v8_lin1_lam10_s1_refs-sched_aobt0_q0_day1_wx1.parquet
run() {
    name=$1; shift
    echo "################ $name: $*"
    if python model_v8.py validate --linear --lin-lambda 10 --day --weather "$@" > "logs/v8_$name.txt" 2>&1; then
        grep -E "===" -A7 "logs/v8_$name.txt"
        python compare_runs.py "$BASE" "$(ls -t logs/pred_v8_*.parquet | head -1)" | head -26
    else
        echo "FAILED, last lines of logs/v8_$name.txt:"
        tail -5 "logs/v8_$name.txt"
    fi
}
{
    run T1 --lr 0.025
    run T2 --leaves 127 --min-leaf 300
    run U1 --aobt
    run U2 --aobt --refs sched,aobt3
    date
} 2>&1 | tee logs/overnight.txt
