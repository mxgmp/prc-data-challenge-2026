# PRC Data Challenge 2026: Taxi-Out Time Prediction

### A validated mixture model for departure taxi-out time at 10 European airports

A machine learning project for the PRC Data Challenge 2026 (team **versatile-kitten**, solo entry), predicting the departure taxi-out time (`TAXITIME_SEC_mvt`, the seconds from off-block to take-off) at 10 European airports. The contest metric is RMSE.

The project started with a simple question:

> How long will a departure taxi before it takes off?

The answer turned out to be more interesting than a regression problem. A large part of the error did not come from taxiing at all, but from how the times were reported: off-block times copied from the schedule, logged on the wrong day, or stored in local time instead of UTC.

The model went through several iterations: an airport and hour average, a gradient-boosted model, a mixture that separates reporting artefacts from real taxis, a rule for a day-error fault at Rome Fiumicino, weather, the NM off-block times once the organisers confirmed they may be used, and finally the current taxi level of other departures and a cost-weighted classifier.

**Final numbers:** clean RMSE 270.92 s on the 2025 validation (all rows 330.13 s) and 305.75 s on the live leaderboard (Jan+Jul 2026). The submitted final file can be reproduced byte for byte from this repository (see [Reproducibility](#reproducibility)).

## The Question

How accurately can departure taxi-out time be predicted at 10 European airports, for months the model has never seen?

The final phase ranks January, February, June and July 2026: Jan+Jul, Feb+Jun and all four months separately. February and June had no leaderboard feedback at all. So the goal is not a good live score but a model that holds up on unseen months, and every decision was made on a validation built to mirror them.

## Dataset

The analysis uses the challenge data:

- movements (arrivals and departures) and the matching Network Manager (NM) flight records for all of 2025 at the 10 airports, used for training
- the ranking files for Jan+Jul 2026 (live leaderboard)
- the final files for Jan, Feb, Jun and Jul 2026

In the ranking and final files, the departure block time and the taxi time are removed.

Weather comes from public METAR reports (see [Data Provenance](#data-provenance)).

Raw data is intentionally excluded from this repository.

## Validation

The final phase scores Jan, Feb, Jun and Jul 2026, while the training data covers 2025. The hold-out therefore uses the same four months of 2025, so seasonal effects (winter de-icing, summer peaks) are tested the way they will be in the final:

- training: Mar-May and Aug-Oct 2025
- early stopping: Nov-Dec 2025
- hold-out: Jan, Feb, Jun and Jul 2025, reported per month pair (Jan+Jul, Feb+Jun) because the organisers rank them separately

The final model is refit on all 12 months of 2025, with the number of trees found by early stopping scaled by 12/6.

Ten hold-out rows have a taxi time over 80,000 s, and they dominate the variance of the all-rows score. "Clean" rows leave them out. Model decisions were made on clean rows, with a bootstrap check (300 resamples, 95% interval of the RMSE difference). Changes to the day-error rule, which only concerns the long rows, were decided on all rows.

Every change followed the same rule:

> A change is kept only if the clean score on all four months is clearly better (the whole interval below zero) and neither month pair gets clearly worse.

The live leaderboard was used only as a sanity check, never for decisions. It scores all rows of two months, where a single day-long taxi time can move the score by several seconds. For example, v10 scored 0.46 s above v9 although it was clearly better on the 2025 clean rows.

## Method

### Features

All features can be computed from the ranking files. Departure block times are removed right after loading, so training rows look exactly like ranking rows. `ARVT_3_flt` is never used.

The features use the flight's own take-off time and the times of other flights around it, both before and after it. For example, `dep_next30` counts departures in the 30 min after this take-off, and the taxi-level features read other departures' take-off and `AOBT_3_flt` times. All of these are fields of the ranking files. A model run before the flight leaves would not have the later times, so this is an after-the-fact estimate, in line with the post-operations use described on the challenge's rationale page.

- **Plan:** take-off minus scheduled time (`sd`), minus EOBT and minus IOBT; differences between the planned times; hour; day of week; missing flight-plan record.
- **NM off-block times:** take-off minus `AOBT_3_flt`, take-off minus `LOBT_flt`, `AOBT_3_flt` minus `LOBT_flt`, and whether `AOBT_3_flt` is missing.
- **Traffic:** departures in the 30 and 60 min before take-off and the 30 min after (`dep_next30`); arrivals within 30 min; departures from the same runway in the previous 30 min.
- **Taxi level of other departures:** for the other departures at the same airport that took off in the 30 and 60 min before this take-off, the mean and median of their take-off minus `AOBT_3_flt` (limited to 0..3600 s; flights without `AOBT_3_flt` skipped) and how many there are. The window ends just before this take-off, so the flight itself is excluded.
- **Recent arrivals:** mean and median taxi-in of arrivals at the same airport in the last 30 and 60 min (clipped at 1 h), and their count.
- **Turnaround:** time since the last arrival on the same stand reached in-block, and that arrival's taxi-in and delay.
- **Scheduled load:** departures scheduled within 30 min of this flight.
- **Weather:** the latest METAR at the departure airport, at most 3 h old: temperature, dew-point spread, wind, gusts, visibility, ceiling, rain, snow, freezing, fog, thunderstorm, de-icing conditions (3 C or colder with precipitation), snow or freezing in the last 6 h, and the report age.
- **Categorical:** airports, runway, stand, aircraft type, wake category, market segment, operator, flight type, flight rule.

### NM Off-Block Fields

`AOBT_3_flt` is the actual off-block time of the flown (M3) trajectory, so it is close to the quantity being predicted. The organisers were asked whether it may be used, and it was kept out of every upload until a written answer arrived. On 9 October 2026 the organisers confirmed in writing (by email and on the challenge Discord) that `AOBT_3_flt` and `LOBT_flt` may be used as inputs, with no changes compared to the initial phase. They are used from v9 on.

Effect on the same validation:

| Inputs (learning rate 0.05) | Clean all 4 | Change | Bootstrap 95% interval |
|---|---|---|---|
| without the fields | 302.83 | | |
| LOBT only | 303.65 | +0.83 | [-0.41, +1.81] |
| AOBT_3 only | 288.87 | -13.96 | [-15.34, -12.47] |
| both | 286.95 | -15.88 | |

At learning rate 0.025 the clean score is 301.43 without the fields and 285.05 with both (-16.38). Almost all of the gain comes from `AOBT_3_flt`.

### The Schedule Artefact and the Mixture Model

At LIRF, 82.9% of departures with a taxi time over 1 h report a taxi time equal to the scheduled delay (take-off minus scheduled time, within 60 s). The off-block time was recorded as the scheduled time, so the reported taxi time includes the delay.

A regressor trained on these rows learns that taxi times can be hours long, and gets worse on normal flights. So the model separates the two cases:

```text
P(artefact) = average of two classifiers:
                LightGBM, multiclass with 2 classes, weight 1 + |max(sd, 0) - normal| / 600
                CatBoost, Logloss
normal      = average of three regressors trained only on non-artefact rows,
              each clipped at 0:
                LightGBM, linear trees:  taxi
                LightGBM, linear trees:  ref + (taxi - ref),  ref = take-off minus AOBT_3 limited to 0..3600 s
                CatBoost:                taxi
prediction  = P(normal) * normal + P(artefact) * max(sd, 0)
```

Under squared error, the expected value is the best single prediction when the outcome is uncertain. A hard switch between the two values is much riskier for RMSE.

The three regressors make different errors, so their average is better than each of them. The second one learns the correction to take-off minus `AOBT_3_flt` instead of the taxi time itself. On its own it is no better than the first, but the average of the two is clearly better (-1.00 s on clean rows). CatBoost builds its trees differently from LightGBM and adds a further -3.02 s.

The same holds for the classifier. Most of the remaining error at LIRF comes from the artefact-or-normal call on delayed flights. Averaging the LightGBM classifier with a CatBoost classifier raises the hold-out AUC at LIRF from 0.863 to 0.867 (all airports 0.938 to 0.947) and improves clean rows by 2.23 s (1 seed, [-2.88, -1.64], measured outside the day-error group so that only the classifier changes), almost all of it at LIRF (-17.3 s).

A wrong artefact call costs about the squared gap between the scheduled delay and the normal taxi, but the classifier's loss counts every flight the same. The LightGBM classifier is therefore trained with sample weight `1 + |max(sd, 0) - normal| / 600`, in training and in the early-stopping set, where `normal` is the in-sample prediction of the two LightGBM regressors (so the regressors are trained first). Its AUC hardly changes (LIRF 0.8628 to 0.8626), but clean rows improve by 1.94 s (1 seed, [-4.04, -0.28]), mostly at LIRF (-13.7 s). The same weights on the CatBoost classifier added nothing.

The artefact label is `|taxi - sd| <= 60 s` at LIRF and `<= 1 s` elsewhere.

### The Day-Error Rule

Sixteen 2025 departures have a taxi time over 80,000 s. Two of them match the scheduled delay (schedule artefacts); the other 14 are day errors. Twelve of the 14 are at LIRF, and for all 12 the taxi time minus one day (86,400 s) is an ordinary taxi of 13 to 33 min.

The likely mechanism: the flight is scheduled on one day and pushes back after midnight, and the airport stores the actual off-block clock time with the scheduled date. The reported taxi time is then the real taxi time plus 86,400 s. This can only happen when the take-off falls on a later local date than the scheduled time, and the data agree:

> All 12 LIRF day errors in 2025 cross local midnight, and none of the about 160,000 LIRF departures that do not cross midnight is a day error.

The two other day errors (Paris CDG and Zurich) do not cross midnight and look like a different, isolated fault.

The rule applies to one group: **LIRF, no flight-plan record, scheduled delay over 4 h, take-off on a later local (Rome) date than the scheduled time.** In 2025 this group has 63 departures, 11 of them day errors (17.46%): 5 of 37 in the non-hold-out months and 6 of 26 in the hold-out months. Eleven cases are too few to learn from, so the rule uses the measured rate:

```text
in the group:  prediction = (1 - p) * mixture prediction + p * (86,400 + normal)
p = 17.46% (all of 2025); in validation p = 13.51%, measured without the hold-out months
```

Day-error rows are removed from the training data of all models.

Before the midnight condition, the group was "LIRF, no flight-plan record, scheduled delay over 4 h" (127 departures, 11 day errors, 8.66%). Adding the condition keeps the same day errors and drops 64 flights that never had one. On the hold-out it catches the same 6 of 8 day errors with 26 instead of 63 flights:

| Day-error group (all rows, all 4 hold-out months) | RMSE | Change | Bootstrap 95% interval |
|---|---|---|---|
| LIRF, no plan, sd over 4 h | 352.53 | | |
| + crosses local midnight | 348.88 | -3.65 | [-7.28, -0.20] |

These numbers are estimated from the saved 1-seed predictions of the two-regressor model, re-weighted with the new group and rate.

### Model Settings

LightGBM with `learning_rate 0.025`, `num_leaves 255`, `min_data_in_leaf 100`, feature and bagging fraction 0.8, `lambda_l2 1`, `deterministic=True`. The two LightGBM regressors use linear trees (`linear_tree`, `linear_lambda 10`): each leaf fits a small linear model, which extrapolates better on long taxis.

CatBoost uses native categorical features, `learning_rate 0.1`, `depth 8`, `l2_leaf_reg 3`, RMSE loss for the regressor and Logloss for the classifier, and early stopping on Nov-Dec after 150 rounds.

The LightGBM classifier uses the cost weights described above; all other models are unweighted. The final model averages 3 seeds. Predictions are capped at the largest 2025 taxi time as a safety limit.

## Results

### Validation (2025, final setup, 3 seeds)

| | all rows | clean |
|---|---|---|
| Jan+Jul | 354.51 | 284.23 |
| Feb+Jun | 302.32 | 256.14 |
| all 4 | 330.13 | 270.92 |

### Ablation (clean rows, all 4 hold-out months, 1 seed)

| Step | Clean RMSE | Change | Bootstrap 95% interval |
|---|---|---|---|
| mixture, linear trees, lambda 10 | 315.39 | | |
| + day-error rule | 307.69 | -7.70 | [-11.60, -3.58] |
| + weather | 302.83 | -4.86 | [-6.88, -2.41] |
| + learning rate 0.025 | 301.43 | -1.39 | [-2.15, -0.77] |
| + NM off-block fields | 285.05 | -16.38 | |
| + second LightGBM regressor relative to AOBT_3 | 284.05 | -1.00 | [-1.33, -0.69] |
| + CatBoost as third regressor | 281.03 | -3.02 | [-3.29, -2.74] |
| + midnight day-error group, 3 seeds | 280.27 | | |
| + CatBoost classifier averaged with the LightGBM classifier, 3 seeds | 276.60 | -3.67 | [-5.38, -2.09] |
| + cost-weighted LightGBM classifier, 1 seed (against 276.34 at 1 seed) | 274.40 | -1.94 | [-4.04, -0.28] |
| + taxi level of other departures, 1 seed | 271.16 | -3.25 | [-4.26, -1.71] |
| both, 3 seeds (final setup, against 276.60) | 270.92 | -5.68 | [-8.45, -3.63] |

Weather helped most where it should: Munich -24.3 s, Paris CDG -15.0 s, Amsterdam -9.3 s on clean rows. CatBoost improved all 10 airports, from -1.8 s at Amsterdam to -5.2 s at Istanbul. The last step also improved all 10 airports, most at Rome (-15.9 s) and Istanbul (-14.7 s). Part of the taxi-level gain comes from one event: a disruption at Istanbul in February 2025, when 3.4% of departures taxied for over 1 h (about 0.1% in other months).

This is the main finding of the project:

> The largest gains came from understanding how the times were recorded and from the right inputs, not from tuning.

The artefact mixture took the 4-month hold-out from 448.74 s to 395.09 s, the day-error rule gave -7.70 s and the NM off-block fields -16.38 s, while the learning rate gave -1.39 s and a different tree size gave nothing.

### Live Leaderboard (Jan+Jul 2026)

| Version | Change | RMSE (s) |
|---|---|---|
| v1 | airport x hour mean | 654.99 |
| v2 | LightGBM, plan and traffic features | 432.98 |
| v3 | artefact mixture | 384.90 |
| v4 | arrival, turnaround and scheduled-load features | 368.20 |
| v5 | linear trees | 337.62 |
| v6 | linear_lambda 10, 3 seeds | 332.50 |
| v7 | day-error rule | 329.66 |
| v8 | weather | not scored (rejected by the daily upload limit) |
| v9 | weather, learning rate 0.025, NM off-block fields, second regressor relative to AOBT_3 | 310.36 |
| v10 | CatBoost as third regressor, midnight day-error group, CatBoost classifier averaged with the LightGBM classifier | 310.82 |
| v11 | taxi level of other departures, cost-weighted LightGBM classifier (final setup) | 305.75 |

v11 is the final setup.

## What Did Not Work

| Idea | Result |
|---|---|
| Cap normal predictions per airport at the 99.9th percentile | removed the linear-tree gain |
| Train the regressor only on taxis under a threshold | rejected on validation |
| More references in the mixture (EOBT, IOBT, AOBT_3) | worse: calibrated on average but cannot tell which flights match |
| Queue features (movements between planned off-block and take-off) | neutral once day errors were removed |
| Wider LIRF artefact window (±5, 10, 15 min) | worse on clean rows (+1.05, +2.54, +6.72 s) |
| 127 leaves, 300 rows per leaf | no gain (+0.52 s) |
| Off-block time minus scheduled time, EOBT and IOBT as extra inputs | no effect (-0.08 s, [-1.05, +0.85]) |
| Separate normal-taxi regressor for LIRF | worse (+0.92 s, [+0.30, +1.47]); LIRF itself +5.3 s |
| Regressor of the correction to take-off minus AOBT_3, alone | no better on clean rows (+0.61 s, [-0.20, +1.36]), worse on all rows; useful only averaged with the direct regressor (see above) |
| Plain LightGBM (no linear trees) as third regressor | worse (+0.18 s, [+0.02, +0.34]) |
| Artefact tolerance outside LIRF of ±15 s or ±30 s | not distinguishable (-0.08 s, [-1.71, +2.06]; +0.49 s, [-1.57, +3.32]); LIRF got worse |
| Third classifier trained only on LIRF flights, averaged in for LIRF rows | worse (+0.69 s, [+0.18, +1.22]); LIRF +4.7 s, although its AUC was slightly higher |
| Day-error group split at 19.4 h, each band with its own smoothed rate | not clearly better than one group (all rows -8.24 s, [-21.00, +0.22]); the long band rests on one 2025 flight |
| Classifier weight 1 + (gap / 600)^2, capped at 100 | not distinguishable (-1.39 s, [-2.84, +0.14]) |
| Taxi level of the departures in the 15 and 30 min after this take-off, added to the final setup | not distinguishable (-0.12 s, [-1.11, +0.76]); better for flights with `AOBT_3_flt`, worse at LIRF |
| Operator taken from the callsign, for flights without a flight-plan record | the patterns are real (some operators are almost always artefacts at LIRF), but the model already captures them: adjustments sized at -0.01 s and +0.12 s, not distinguishable |
| Average of the v10 and v11 models | worse than v11 alone (+2.04 s, [+1.23, +3.26]) |

## Findings Not Used

- At LIRF, some departures without a flight-plan record and with a small scheduled delay report a taxi time of a whole number of hours plus a normal taxi, where the hours equal Rome's UTC offset (1 h in winter, 2 h in summer): 24 flights in 2025, 12 in the hold-out months and 12 in the others. It looks like a local time stored as UTC, and it is concentrated in one operator's flights. A rule hedging for it would gain about 0.2 s, and its cut-off would have been chosen on the hold-out, so it is not used.
- The February 2025 Istanbul disruption: in congested periods, flights without a flight-plan record were under-predicted in the hold-out, but outside the hold-out they behave like other flights. A feature for it would fit a single event, so none was added.

## Reproducibility

Python 3.14.6 with the exact package versions in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Put the challenge files in `data/raw/`: the 12 `training_2025-*.parquet` files, `ranking.parquet`, `submitting.parquet`, `final_ranking.parquet`, `final_submitting.parquet`.
2. Weather: `python fetch_metar.py download`, then `python fetch_metar.py build`. This writes `data/external/metar.parquet`. A new download on 11 Oct 2026 gave files identical to the ones used for the final model.
3. Validate: `python taxiout.py validate --aobt --seeds 3`. This prints the scores above and saves the number of trees in `models/`. About 2 h (126 min) on 16 cores, most of it CatBoost.
4. Submit: `python taxiout.py submit --aobt --seeds 3 --version N`. This writes `submissions/versatile-kitten_vN.parquet` (leaderboard) and `submissions/versatile-kitten_final_candidate_vN.parquet` (final phase). About 7.3 h on 16 cores, most of it CatBoost. `python check_final.py N` checks the final file against `final_submitting.parquet`.

The whole chain was verified on 11 October 2026 in a fresh folder containing only the files of this repository, a new virtual environment built from `requirements.txt`, the raw challenge data and the downloaded METAR files. It reproduced the validation predictions row for row and wrote a final file byte-identical to the submitted one (`versatile-kitten_final.parquet`, 670,790 rows, md5 `7ad4e280932a1be0177bbcc9df0d0a74`). The full run took about 9.4 h on 16 cores.

With `deterministic=True` the results repeat exactly on the same machine and library versions. Other CPUs or versions can give small differences.

## Repository Layout

- **Pipeline:** `taxiout.py` (features, models, validation and submission), `fetch_metar.py` (weather data), `check_final.py` (checks a final file against the template), `upload_final.py` (the one-time final upload), `prc_fetch_and_check.py` (downloads the challenge data), `scores.py` (live scores).
- **Analysis:** `compare_runs.py` compares two saved validation runs row by row with the bootstrap check, and `compare_pairs.py` gives the interval for each month pair. `blend_check.py` does the same for averages of saved runs. `error_budget.py` breaks the error down by airport and taxi-time band. `diag_midnight.py` and `diag_dayrule.py` reproduce the day-error analysis; the other `diag_*.py`, `compare_k.py`, `compare_m.py` and `null_split.py` are the diagnostics described in `NOTEBOOK.md`.
- **Experiments:** each `mk_exp_*.py` writes its `exp_*.py` by patching `taxiout.py`, and its docstring says which version of `taxiout.py` it was written for. Most stop without writing anything on the final `taxiout.py`; `mk_exp_m2.py` and `mk_exp_t.py` still apply to it. `exp_q.py` and `exp_k2.py` reuse saved runs instead of patching.
- **History:** `baseline_v1.py` and `model_v2.py` to `model_v8.py` are the earlier versions behind leaderboard uploads v1 to v8. `today.sh`, `next.sh`, `overnight.sh` and `batch2.sh` are the batch scripts that ran the experiments.
- **Documentation:** this README and `NOTEBOOK.md`, the experiment log with every idea, its numbers and the decision, including the failures.

Scripts that talk to the challenge buckets load the credentials from a local file outside the repository (`~/.config/prc/minio.env`). No credentials are stored here.

## Data Provenance

The challenge data is provided by the organisers of the PRC Data Challenge 2026 and is not redistributed here.

Weather comes from METAR reports in the Iowa Environmental Mesonet (IEM) archive, Iowa State University, downloaded by `fetch_metar.py` from:

`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`

The IEM states that its materials are in the public domain and may be used freely by anyone for any lawful purpose (`https://mesonet.agron.iastate.edu/disclaimer.php`). Period: 2025-01-01 to 2026-08-01, the 10 departure airports. This repository does not redistribute the weather data.

## Limitations

The day-error rule relies on a 2025 rate measured on 63 flights (11 day errors). If the fault changes in 2026, the rule will be miscalibrated.

LIRF remains the largest source of error: clean RMSE 509 s, against 176 to 335 s at the other airports. Flights without their own `AOBT_3_flt` (1.3% of rows, almost all without a flight-plan record) cause about 29% of the clean squared error (1-seed run).

The validation uses 2025 months, not 2026. Year-to-year changes in traffic or procedures are not covered.

The same Jan and Jul 2026 flights carry different movement and block times in `ranking.parquet` and `final_ranking.parquet` (up to ±12 s, median 4 s), so predictions for those months differ slightly between the leaderboard file and the final file (v11: RMS 11.7 s, median 3.1 s, 886 of 344,841 flights over 60 s).

The 2025 training files carry the same kind of noise on block times (about ±6 s on each file, which gives the ±12 s between two files). The ±1 s artefact label outside LIRF therefore catches only about a quarter of the artefacts there; a wider label did not help (see above).

## Project Status

**Final submission uploaded on 12 October 2026.**

The final model (v11) is frozen. The project demonstrates a full validation-driven workflow: a hold-out that mirrors the final months, a fixed decision rule with bootstrap intervals, documented failures, and a submission that can be reproduced byte for byte.

The next step, once the results are out, is to compare this approach with the top solutions and see where the remaining error could be reduced, especially the artefact-or-normal call at LIRF.

## License

GNU General Public License v3.0. See `LICENSE`.
