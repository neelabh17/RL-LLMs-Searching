from datasets import load_dataset
from huggingface_hub import HfApi
from datasets import DatasetDict, Dataset

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
correct_ds_name  = "neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr"
percentages = [0, 20, 40, 60, 80]
correct_truncated_datasets = {}
for split in ["train", "test"]:
    correct_dataset = load_dataset(correct_ds_name, split=split)  # or load_dataset("json", data_files="file.json")


    for pct in percentages:
        truncated = correct_dataset.map(lambda ex: truncate_text(ex, pct))
        if f"{correct_ds_name}_{pct}" not in correct_truncated_datasets:
            correct_truncated_datasets[f"{correct_ds_name}_{pct}"] = {}
        correct_truncated_datasets[f"{correct_ds_name}_{pct}"][split] = truncated

for name in correct_truncated_datasets:
    # breakpoint()
    dataset_final = DatasetDict({
            "train": correct_truncated_datasets[name]["train"],
            "test": correct_truncated_datasets[name]["test"]
        }) 
    hf_access_token = "sample_huggingface_token"
    dataset_final.push_to_hub(name, token=hf_access_token)

