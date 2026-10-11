# Appendix A: Competition-specific post-processing (considered, implemented as options, not used in the submission)

Two post-processing rules were considered. Both use recording artefacts in the LIRF (Rome Fiumicino) departure records of this dataset; they are not modelling methods and we do not expect them to transfer to other data. They are implemented as options, are disabled, and are **not part of the submitted predictions**.

## Where the code is

- `pipeline/finalize/postprocessing_options.py` (subcommand `apply`, one on/off switch per rule) and its driver `pipeline/drivers/run_postprocessing_options.sh` (one on/off environment switch per rule; names in the driver header). The switches have no default; both must be given explicitly. In our run, both switches off reproduced the submitted file byte for byte (step 10 of `pipeline/drivers/run_order.sh`). The submitted file is the output of the submission chain itself (step 9) and does not pass through this step.
- Reference implementation of rule 1: `experiments/not_adopted/lirf_airline_rule.py`.
- 2025 evaluation of both rules: `evaluation/cv_recheck_design_choices.py`.

The rules would sit after the partial-day patch and before the lower bound, which is re-applied after them. Their hit sets are disjoint (rule 1 requires a network-manager off-block time, rule 2 its absence).

## Rule 1: LIRF airline artefact rule

- **Candidates:** LIRF departures with a network-manager actual off-block time (AOBT) and take-off − scheduled time exceeding take-off − AOBT by more than 1,200 s.
- **Groups:** for each airline, and each airline × destination, the share of 2025 training candidates whose label equals take-off − scheduled time within 60 s (the block time was recorded as the scheduled time). A group qualifies with at least 30 candidates and a share of at least 0.8.
- **Action:** candidates in a qualifying group are predicted as `rint(take-off − scheduled time)`.
- **2025 CV** (group shares estimated leave-one-month-out, 5 day folds over January and July 2025): ΔRMSE −0.236 s, ΔMSE −139 ± 42 (day-block SE), 5 of 5 folds improve.

## Rule 2: LIRF extreme-row rule

- **Hits:** LIRF departures without a network-manager off-block time and with take-off − scheduled time `ms` above 36,000 s.
- **Action:** if `ms ≤ 50,000`, predict `ms`; otherwise predict `rint((1 − P) · ms + P · 87,300)`, where `P` is the share of 2025 rows of the same kind with `ms > 50,000` whose label is at least 86,400 s and differs from `ms` by more than 600 s (P = 0.55 on all of 2025, 20 rows; in CV estimated from the other 11 months).
- **2025 CV:** ΔRMSE −6.820 s, ΔMSE −3,975 ± 2,237 (day-block SE), 4 of 5 folds improve.

## Why they are not part of the method

Both rules satisfy our default CV rule (pooled ΔSSE ≤ 0 and at least 4 of 5 day folds improve). We still keep them out of the submission, because the submitted predictions should correspond one to one to the method in the README, and these rules are dataset-specific post-processing rather than modelling. In addition, from 2025 data alone:

- **Rule 2** is a bet on a handful of records: the whole gain comes from 14 rows, and the standard error is more than half of the gain. It assumes that the 2025 distribution of these extreme records holds in 2026; a small change turns the gain into a loss, and each hit row is large enough to move the overall RMSE.
- **Rule 1** gains only 0.236 s in CV, and the artefact it relies on (block time recorded as the scheduled time for some airlines) cannot be checked in the test period, where block times are not given.
