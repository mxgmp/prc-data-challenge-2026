#!/usr/bin/env bash
# model_v8.py: LOBT, AOBT_3 and learning-rate runs V1 to V3 (NOTEBOOK entry 27). Run: bash batch2.sh
P=logs/pred_v8_lin1_lam10_s1_refs-sched
BASE=${P}_aobt0_q0_day1_wx1.parquet
run() {
    name=$1; shift
    echo "################ $name: $*"
    if python model_v8.py validate --linear --lin-lambda 10 --day --weather "$@" > "logs/v8_$name.txt" 2>&1; then
        grep -E "===" -A7 "logs/v8_$name.txt"
        python compare_runs.py "$BASE" "$(ls -t logs/pred_v8_*.parquet | head -1)"
    else
        echo "FAILED, last lines of logs/v8_$name.txt:"
        tail -5 "logs/v8_$name.txt"
    fi
}
{
    run V1 --aobt --obt lobt
    run V2 --aobt --obt aobt3
    run V3 --aobt --lr 0.025
    echo "################ T1 (lr 0.025) vs base, clean rows"
    python compare_runs.py "$BASE" "${P}_aobt0_q0_day1_wx1_lr0.025.parquet" | grep "'clean all 4'"
    echo "################ U2 (mixture with AOBT_3) vs U1 (features only)"
    python compare_runs.py "${P}_aobt1_q0_day1_wx1.parquet" "${P}-aobt3_aobt1_q0_day1_wx1.parquet" | grep -E "^clean all 4|bootstrap"
    date
} 2>&1 | tee logs/batch2.txt
