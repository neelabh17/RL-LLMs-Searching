# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import compute_0_1_score
import torch
from tqdm import tqdm, trange
import numpy as np


class ValidationRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, experiment_name, project_name, prefix, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or compute_0_1_score
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.prefix = prefix

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            #extract the prefix
            # prefix = prompt_str.split("")[-1]
            # print("[prompt]", prompt_str)
            # print("[response]", response_str)
            # print("-" * 40)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        scores_per_id = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem


            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # print("[prompt]", prompt_str)
            # print("[response]", response_str)
            # print("-"*40)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            index = data_item.non_tensor_batch['extra_info']["index"]


            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            # print("Index:", index)  
            if index not in scores_per_id:
                scores_per_id[index] = []
            scores_per_id[index].append(score)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        # Calculatin the pass@k metric
        total_indices = len(scores_per_id)
        solved = 0
        total_solved = 0
        for index in scores_per_id:
            scores = scores_per_id[index]
            np_scores_array = np.array(scores)
            print(np_scores_array)
            if np.sum(np_scores_array) > 0:
                solved += 1
            total_solved += np.mean(np_scores_array)
        import os
        save_path = os.path.join("evals", self.project_name, self.experiment_name + "_" + self.prefix)
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "scores.txt"), "w") as f:
            print(len(scores_per_id), "indices", file=f)
            print("number of generations:", len(scores_per_id[index]), file=f)
            pass_at_k = solved / total_indices
            print(f"Pass@100: {pass_at_k}", file=f)
            print("Fraction of problems got right from top 100:", total_solved / total_indices, file=f)
        return reward_tensor
