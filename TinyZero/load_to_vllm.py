import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM  # or appropriate class

# Step 1: Load the config
load_path = "checkpoints/TinyZero/1.5B_partial_corr_80/global_step_100/actor"
config = AutoConfig.from_pretrained(os.path.join(load_path, "huggingface"))

# Step 2: Initialize model from config
model = AutoModelForCausalLM.from_config(config)

# Step 3: Load the PyTorch weights
state_dict = torch.load(os.path.join(load_path, "model_world_size_1_rank_0.pt"), map_location="cpu")
# breakpoint()

# Step 4: Load weights into model
model.load_state_dict(state_dict)
save_path = load_path.replace("checkpoints", "vllm_ready_checkpoints")

# Step 5: Save the model
model.save_pretrained(save_path)
config.save_pretrained(save_path)

print("Model loaded successfully!")