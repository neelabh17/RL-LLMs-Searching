''''
To load the traces from the 14B model and for each sample selec 1 correct and 1 wrong trace at random and create a dataset of that and use it for partial inference
'''
from datasets import load_dataset
from datasets import DatasetDict, Dataset
from tqdm import trange

traces_dataset_name = "bicycleman15/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct"
traces_dataset = load_dataset(traces_dataset_name, split="train")  # or load_dataset("json", data_files="file.json")


correct_list = []
incorrect_list = []
correct_index_list = {}
incorrect_index_list = {}
for i in trange(len(traces_dataset)):
    atleast_one_correct = False
    atleast_one_incorrect = False
    for j in range(128):
        if traces_dataset[i][f"correct_{j}"] == 1:
            atleast_one_correct = True 
            correct_index_list[i] = j
        if traces_dataset[i][f"correct_{j}"] == 0:
            atleast_one_incorrect = True 
            incorrect_index_list[i] = j
    correct_list.append(atleast_one_correct)
    incorrect_list.append(atleast_one_incorrect)

import numpy as np
correct_list = np.array(correct_list)
incorrect_list = np.array(incorrect_list)

assert np.sum(correct_list) == len(traces_dataset), "Not all samples have at least one correct trace"
assert np.sum(incorrect_list) == len(traces_dataset), "Not all samples have at least one correct trace"
correct_response_list = []
incorrect_response_list = []
indices = np.arange(len(traces_dataset)).tolist()
for i in trange(len(traces_dataset)):
    correct_response_list.append(traces_dataset[i][f"response_{correct_index_list[i]}"])
    incorrect_response_list.append(traces_dataset[i][f"response_{incorrect_index_list[i]}"])

correct_ds = Dataset.from_dict({
    "index": indices,
    "graph": traces_dataset["graph"],
    "source": traces_dataset["source"],
    "destination": traces_dataset["destination"],
    "path": traces_dataset["path"],
    "response": correct_response_list,
    })

incorrect_ds = Dataset.from_dict({
    "index": indices,
    "graph": traces_dataset["graph"],
    "source": traces_dataset["source"],
    "destination": traces_dataset["destination"],
    "path": traces_dataset["path"],
    "response": incorrect_response_list,
    })
hf_access_token = "sample_huggingface_token"
correct_ds.push_to_hub("neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_corr", token=hf_access_token)
incorrect_ds.push_to_hub("neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_incorr", token=hf_access_token)

# breakpoint()
print("done")




