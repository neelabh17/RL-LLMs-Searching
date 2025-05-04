# For the partial trace inference experiment
- We took dataset from the 14B traces at `bicycleman15/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct` and selected one correct (1C) and one incorrect (1IC) generation for each row and create:
- `neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_corr` and `neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_incorr` dataset
- These 1C, 1IC are used to create datsaets that have partial traces from the 14B model at: 
    - `neelabh17/incorr_partial_40_num_gen_100_Qwen2.5-1.5B-Instruct`
    - `neelabh17/corr_partial_40_num_gen_100_Qwen2.5-1.5B-Instruct`

with various `partial_percentages = [20, 40, 60, 80]`, `incorr` and `corr`, and `sizes = [0.5B, 1.5B, 3B]`

All the results can be found for this evaluation with metric of `pass@100` in `evals/eval_partial_traces`

We shall also evaluate out of the box `pass@100` performance of various model sizes and can be found at `evals/eval_out_of_the_box_traces` 

They are present at `neelabh17/corr_out_of_the_box_num_gen_100_Qwen2.5-0.5B-Instruct`



### Pass@100 (Avg. Correct per Sample) for Correct and Incorrect Traces from 14B Model

| **% Trace**           | **Correct Traces (0.5B)** | **Correct Traces (1.5B)** | **Correct Traces (3B)** | **Incorrect Traces (0.5B)** | **Incorrect Traces (1.5B)** | **Incorrect Traces (3B)** |
|-----------------------|---------------------------|----------------------------|--------------------------|------------------------------|-----------------------------|----------------------------|
| **20%**              | 0 (0)                     | 36 (0.66)                  | 56 (3.32)                | 2 (0.02)                     | 25 (0.43)                   | 45 (2.41)                  |
| **40%**              | 2 (0.02)                  | 43 (1.14)                  | 60 (3.51)                | 0 (0)                        | 17 (0.42)                   | 50 (2.48)                  |
| **60%**              | 2 (0.02)                  | 47 (1.73)                  | 74 (6.02)                | 2 (0.02)                     | 24 (0.84)                   | 46 (3.54)                  |
| **80%**              | 12 (1.19)                 | 58 (9.02)                  | 79 (18.29)               | 3 (0.56)                     | 22 (1.63)                   | 47 (4.11)                  |
| **Out of the box**   | **3 (0.03)**              | **31 (0.64)**              | **76 (3.78)**            | **3 (0.03)**                 | **31 (0.64)**               | **76 (3.78)**              |

---
# Trying out RLFT on partial completions

## Requirements
- Completions from 14B on the train and the test set so that inference can be done in line with the training procedule which is `prompt + prefix` 

Procedure 
1. We took the OG dataset from `anirudhb11/star-graph-deg-5-path-5-nodes-300` and ran 10 commpletions from the `Qwen14B-Instruct` model for 5k train samples and 100 test samples at `neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct` using the file `helpers/create_traces_DS.py`

We get about 98 test queries having atleast 1 correct completion and 4962 of train samples having atleast 1 correct completion.

We save this dataset at `neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr` using the file `helpers/create_1_corr_train_DS_from_partial_DS.py`

We create partial responses datasets with `[0, 20, 40, 60, 80]` and save at `neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_{pct}`

We shall now run RLFT on these datasets