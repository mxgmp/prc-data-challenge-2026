#!/usr/bin/env bash
# Experiments L and R against the 1-seed taxiout.py run (NOTEBOOK entries 32, 33). Run: bash next.sh
BASE=logs/pred_taxiout_s1_aobt.parquet
run() {
    name=$1; script=$2; tagname=$3
    echo "################ $name"
    if python "$script" validate --aobt > "logs/${tagname}.txt" 2>&1; then
        grep -A9 "===" "logs/${tagname}.txt"
        python compare_runs.py "$BASE" "logs/pred_${tagname}.parquet"
    else
        echo "FAILED, last lines of logs/${tagname}.txt:"
        tail -5 "logs/${tagname}.txt"
    fi
}
{
    python mk_exp_l.py
    python mk_exp_r.py
    run "L: separate normal-taxi model for LIRF" exp_l.py expl_s1_aobt
    run "R: normal taxi as a correction to take-off minus AOBT_3" exp_r.py expr_s1_aobt
    date
} 2>&1 | tee logs/next.txt
