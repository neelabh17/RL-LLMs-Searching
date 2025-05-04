
import os
import argparse
import datetime
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset


# dataset_name =  "neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_100_Qwen2.5-7B-Instruct"
# dataset_name =  "neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_100_Qwen2.5-14B-Instruct"
dataset_name =  "neelabh17/star-graph-deg-10-path-10-nodes-300_out_of_the_box_num_gen_100_Qwen2.5-14B-Instruct"
print(dataset_name)
# load HF dataset
dataset = load_dataset(dataset_name, split="train")
# dataset = dataset.select(range(100))  # use the train split only

colums = dataset.column_names
num_generations = int(colums[-1].split("_")[-1]) + 1

correct = 0
num_correct = []
for i in range(len(dataset)):
    correct_in_a_row = 0
    for j in range(num_generations):
        if dataset[i][f"correct_{j}"] == 1:
            correct_in_a_row += 1
    if correct_in_a_row > 0:
        correct += 1
    num_correct.append(correct_in_a_row)
num_correct = np.array(num_correct)
mean_correct = np.mean(num_correct)
    
save_path = os.path.join("evals", "eval_out_of_the_box_traces", dataset_name)
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, "eval.txt"), "w") as f:
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Mean correct: {mean_correct}\n")
    f.write(f"Num correct: {correct}\n")
    f.write(f"Num generations: {num_generations}\n")
    f.write(f"Num samples: {len(dataset)}\n")


# breakpoint()
print(colums)