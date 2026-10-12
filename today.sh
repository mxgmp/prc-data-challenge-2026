#!/usr/bin/env bash
# Reproduction check of taxiout.py against run V3, then experiment W (NOTEBOOK entries 30, 31). Run: bash today.sh
V3=logs/pred_v8_lin1_lam10_s1_refs-sched_aobt1_q0_day1_wx1_lr0.025.parquet
{
    echo "################ 1. taxiout.py with AOBT, 1 seed, must reproduce run V3"
    python taxiout.py validate --aobt > logs/taxiout_s1_aobt.txt 2>&1
    grep -A9 "===" logs/taxiout_s1_aobt.txt
    python -c "
import pandas as pd
a = pd.read_parquet('$V3')
b = pd.read_parquet('logs/pred_taxiout_s1_aobt.parquet')
same = (a.MVT_ID_mvt.values == b.MVT_ID_mvt.values).all() and (a.pred.values == b.pred.values).all()
print('reproduction check:', 'IDENTICAL' if same else 'DIFFERENT')
"
    echo "################ 2. W: off-block time minus scheduled, EOBT and IOBT as extra inputs"
    python mk_exp_w.py
    python exp_w.py validate --aobt > logs/expw_s1_aobt.txt 2>&1
    grep -A9 "===" logs/expw_s1_aobt.txt
    python compare_runs.py logs/pred_taxiout_s1_aobt.parquet logs/pred_expw_s1_aobt.parquet
    date
} 2>&1 | tee logs/today.txt
