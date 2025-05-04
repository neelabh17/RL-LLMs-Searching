''''
To load the traces from the 14B model and for each sample selec 1 correct and 1 wrong trace at random and create a dataset of that and use it for partial inference
'''
from datasets import load_dataset
from datasets import DatasetDict, Dataset
from tqdm import trange

traces_dataset_name = "neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct"
splits = ["train", "test"]
final_dataset = {}
for split in splits:
    traces_dataset = load_dataset(traces_dataset_name, split=split)  # or load_dataset("json", data_files="file.json")
    num_gens = 0
    for col in traces_dataset.column_names:
        if "answer_" in col:
            num_gens += 1
    
    


    correct_indices = []
    correct_answer_index_list = []
    for i in trange(len(traces_dataset)):
        atleast_one_correct = False
        for j in range(num_gens):
            if traces_dataset[i][f"correct_{j}"] == 1:
                atleast_one_correct = True 
                correct_indices.append(i)
                correct_answer_index_list.append(j)
                break



    import numpy as np
    correct_indices = np.array(correct_indices)

    # shorten traces_dataset to only include the correct indices
    traces_dataset_shortened = traces_dataset.select(correct_indices.tolist())
    assert len(traces_dataset_shortened) == len(correct_indices), "Not all samples have at least one correct trace"

    correct_response_list = []
    for i in trange(len(traces_dataset_shortened)):
        correct_response_list.append(traces_dataset_shortened[i][f"response_{correct_answer_index_list[i]}"])

    final_dataset[split] = Dataset.from_dict({
        "index": traces_dataset_shortened["index"],
        "graph": traces_dataset_shortened["graph"],
        "source": traces_dataset_shortened["source"],
        "destination": traces_dataset_shortened["destination"],
        "path": traces_dataset_shortened["path"],
        "response": correct_response_list,
        })


dataset_final = DatasetDict({
            "train": final_dataset["train"],
            "test": final_dataset["test"]
        }) 
hf_access_token = "sample_huggingface_token"
dataset_final.push_to_hub("neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr", token=hf_access_token)

# breakpoint()
print("done")




