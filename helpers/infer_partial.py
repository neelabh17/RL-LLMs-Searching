"""
Sample command:

python infer_partial.py \
--dataset_name "neelabh17/star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_1_incorr" \
--model_name "Qwen/Qwen2.5-1.5B-Instruct" \
--perctentage_partial 20 \
--temperature 0.8 \
--top_p 0.95 \
--max_tokens 4096 \
--num_generations 10 --push_to_hub

"""

import os
import argparse
import datetime
import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset


def create_prompts(dataset, tokenizer):
    BASE_PROMPT = open("sft/prompts/SFT_v0.md", "r").read()
    prompts = list()
    for i in range(len(dataset)):
        # print(BASE_PROMPT)
        # print(dataset)
        # breakpoint()
        prompt = BASE_PROMPT.format(graph=dataset[i]["graph"], source=dataset[i]["source"], destination=dataset[i]["destination"])

        # apply chat template to ensure right formatting
        prompt = [{"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
                  {"role": "user", "content": prompt},
                  {"role": "assistant", "content": f"Let me solve this step by step.\n<think>\n {dataset[i]['response']}"}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False, continue_final_message = True)
        prompts.append(prompt)
        if i < 5:
            print(prompt)
    return prompts

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    answer = answer.replace(" ", "") # remove spaces too!
    return answer.strip()

def create_results_folder(dataset_name, model_name):
    # create datetime
    # name = str(datetime.datetime.now())
    # name = "/".join(name.split(" ")) # day/time
    name = ""
    name += f"{dataset_name.split('/')[-1]}_{model_name.split('/')[-1]}"
    return name

if __name__ == "__main__":

    # create argparser
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="neelabh17/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct-final-corr")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--perctentage_partial", type=int, default=20)
    # parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--output_folder", type=str, default="Results")
    parser.add_argument("--push_to_hub", action="store_true")

    args = parser.parse_args()
    print(args)

    # load HF dataset
    final_ds = args.dataset_name + f"_{args.perctentage_partial}"
    dataset = load_dataset(final_ds, split="train") # use the train split only

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = create_prompts(dataset, tokenizer)

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, n=args.num_generations)
    llm = LLM(model=args.model_name)

    # responses = llm.generate(prompts[:10], sampling_params, use_tqdm=True)
    responses = llm.generate(prompts, sampling_params, use_tqdm=True)

    # dump outputs to a file
    saving_ds_name = f"{args.dataset_name.split('_')[-1]}_partial_{args.perctentage_partial}_num_gen_{args.num_generations}"
    results_folder = os.path.join(args.output_folder, create_results_folder(saving_ds_name, args.model_name))
    print("Saving generation to: ", results_folder)
    os.makedirs(results_folder, exist_ok=True)

    torch.save(responses, os.path.join(results_folder, "responses.pt")) # binarize the outputs and save

    if args.push_to_hub:
        #### push to HF hub
        print("Pushing to HF Hub...")

        json_responses = list()
        for i in range(len(responses)):

            temp = {}
            temp["index"] = dataset[i]["index"]
            temp["graph"] = dataset[i]["graph"]
            temp["source"] = dataset[i]["source"]
            temp["destination"] = dataset[i]["destination"]
            temp["question"] = prompts[i]
            temp["path"] = dataset[i]["path"]

            # push the outputs
            for j, x in enumerate(responses[i].outputs):
                temp[f"response_{j}"] = x.text

                # we can also extract the answer too
                temp[f"answer_{j}"] = extract_xml_answer(x.text)
                temp[f"correct_{j}"] = 1 if temp[f"answer_{j}"] == dataset[i]["path"] else 0

            json_responses.append(temp)

        response_data = Dataset.from_list(json_responses)
        
        response_data.save_to_disk(results_folder)

        # create dataset name
        _splits = results_folder.split("/")
        # hf_dataset_name = "_".join([_splits[1], _splits[2]])
        hf_dataset_name = _splits[-1]
        hf_dataset_name = hf_dataset_name.replace(":", ".")
        hf_dataset_name = f"neelabh17/{hf_dataset_name}"
        print("Pushing to HF Hub as: ", hf_dataset_name)

        hf_access_token = "sample_huggingface_token"
        response_data.push_to_hub(hf_dataset_name, token=hf_access_token)

    print("Done!")