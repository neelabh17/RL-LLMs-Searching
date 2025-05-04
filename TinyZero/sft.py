# LORA Training, adapted from https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py
# All: anirudhb11/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_instruct_all
# Only correct: anirudhb11/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_flattened_corr
# Change the output dir as per your convenience.
"""
python sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name anirudhb11/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_instruct_all \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 4096 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy no \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir outputs \
    --push_to_hub
"""


import argparse

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

def formatting_func(example):
    text = f'{example["question"]}\n{example["answer"]}'
    return text


# ALL_DATASET_NAME -- 'anirudhb11/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_flattened_all'
# CORRECT_DATASET_NAME -- 'anirudhb11/2025-03-10_23.42.26.740895_star-graph-deg-5-path-5-nodes-300-Qwen2.5-14B-Instruct_flattened_corr'

def main(script_args, training_args, model_args):
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    print('Model Kwargs: ', model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    response_template = '<|im_start|>assistant\nLet me solve this step by step.\n<think>'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        formatting_func = formatting_func, # New line separated question and answer
        data_collator=collator
    )
    
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)



def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)