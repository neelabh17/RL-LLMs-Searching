"""
Preprocess dataset for Tree Search Dataset task
"""

import re
import os
from datasets import load_dataset
import argparse


def make_prefix(dp, template_type):
    graph = dp['graph']
    source_node_indx = dp['source']
    destination_node_indx = dp['destination']
    response = dp["response"]


    if template_type == 'base':
        # This is the wrong way top do things
        # """This works for any base model"""
#         prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
# User: Given a graph in the form of '|' separated edges, output a path from source node to the destination node in the form of comma separated integers. For this question the graph is {graph}\nThe source node is {source_node_indx}\nThe destination node is {destination_node_indx}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 14,2,3 </answer>.
# Assistant: Let me solve this step by step.
# <think>"""
        pass
    elif template_type == 'qwen-instruct':
        prompt = f"Given a graph in the form of '|' separated edges, output a path from source node to the destination node in the form of comma separated integers. For this question the graph is {graph}\nThe source node is {source_node_indx}\nThe destination node is {destination_node_indx}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 14,2,3 </answer>."
        prompt = [{"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
                  {"role": "user", "content": prompt},
                  {"role": "assistant", "content": f"Let me solve this step by step.\n<think>\n {response}"}]
        # """This works for Qwen Instruct Models"""
        # prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Given a graph in the form of '|' separated edges, output a path from source node to the destination node in the form of comma separated integers. For this question the graph is {graph}\nThe source node is {source_node_indx}\nThe destination node is {destination_node_indx}. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 14,2,3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>\n{response}"""
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/gpfs/data/ranganathlab/Jatin/verl/data')
    parser.add_argument('--template_type', default='qwen-instruct')
    parser.add_argument('--dataset_name', default='anirudhb11/star-graph-deg-5-path-5-nodes-300')

    args = parser.parse_args()

    data_source = 'tree_search_partially_completed'

    train_dataset = load_dataset(f'{args.dataset_name}', split='train')
    test_dataset = load_dataset(f'{args.dataset_name}', split='test')

    # truncate dataset to 5000 entries only
    # train_dataset = train_dataset.select(range(5000))
    # test_dataset = test_dataset.select(range(100))

    # breakpoint()

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            # We were not using this at all so commented it out
            # solution = {
            #     "path": example['path'],
            # }
            data = {
                "data_source": data_source,
                "prompt": question,
                "ability": "tree_search_partially_completed",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, args.dataset_name, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, args.dataset_name, 'test.parquet'))

'''
Run as 
python examples/data_preprocess/tree_search_partially_completed.py --local_dir /gpfs/data/ranganathlab/Jatin/verl_neel/data --template_type qwen-instruct --dataset_name neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_0
python examples/data_preprocess/tree_search_partially_completed.py --local_dir /gpfs/data/ranganathlab/Jatin/verl_neel/data --template_type qwen-instruct --dataset_name neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_20
python examples/data_preprocess/tree_search_partially_completed.py --local_dir /gpfs/data/ranganathlab/Jatin/verl_neel/data --template_type qwen-instruct --dataset_name neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_40
python examples/data_preprocess/tree_search_partially_completed.py --local_dir /gpfs/data/ranganathlab/Jatin/verl_neel/data --template_type qwen-instruct --dataset_name neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_60
python examples/data_preprocess/tree_search_partially_completed.py --local_dir /gpfs/data/ranganathlab/Jatin/verl_neel/data --template_type qwen-instruct --dataset_name neelabh17/star-graph-deg-5-path-5-nodes-300_out_of_the_box_num_gen_10_Qwen2.5-14B-Instruct_1_corr_80
ok
'''