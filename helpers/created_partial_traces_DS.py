from datasets import load_dataset

def truncate_text(example, pct):

    text = example["response"]
    try:
        text = text.split("</think>")[0]
        cut_length = max(1, int(len(text) * pct / 100))
        return {"response": text[:cut_length]}
    except:
       print("Error in truncating text")
       cut_length = max(1, int(len(text) * pct / 100))
       return {"response": text[:cut_length]}


# Load your dataset from HF or local
correct_ds_name  = "neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_corr"
correct_dataset = load_dataset(correct_ds_name, split="train")  # or load_dataset("json", data_files="file.json")
incorrect_ds_name  = "neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_incorr"
incorrect_dataset = load_dataset(incorrect_ds_name, split="train")  # or load_dataset("json", data_files="file.json")

percentages = [20, 40, 60, 80]
correct_truncated_datasets = {}
incorrect_truncated_datasets = {}

for pct in percentages:
    truncated = correct_dataset.map(lambda ex: truncate_text(ex, pct))
    correct_truncated_datasets[f"{correct_ds_name}_{pct}"] = truncated

    truncated = incorrect_dataset.map(lambda ex: truncate_text(ex, pct))
    incorrect_truncated_datasets[f"{incorrect_ds_name}_{pct}"] = truncated




from huggingface_hub import HfApi
from datasets import DatasetDict, Dataset

api = HfApi()

for name, dset in correct_truncated_datasets.items():
    correct_dataset_dict = DatasetDict({
        "train": dset
    })
    correct_dataset_dict.push_to_hub(f"neelabh17/{name.strip('neelabh17/')}")

for name, dset in incorrect_truncated_datasets.items():
    incorrect_dataset_dict = DatasetDict({
        "train": dset
    })
    incorrect_dataset_dict.push_to_hub(f"neelabh17/{name.strip('neelabh17/')}")


# Run as 
# python helpers/created_partial_traces_DS.py