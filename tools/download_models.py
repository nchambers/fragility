from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


models = [ #"microsoft/Phi-3-medium-4k-instruct"
    #"microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-small-8k-instruct", "microsoft/Phi-3-medium-4k-instruct",
    #"google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it",
    #"Qwen/Qwen3-0.6B"#, "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"
    #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-0325-32B-Instruct"
]


# Command-line argument.
import argparse
parser = argparse.ArgumentParser(description="Download models to disk.")
parser.add_argument("dirpath", type=str, help="Path to the directory to save the model subdirectories.")
args = parser.parse_args()
local_dir = args.dirpath


# Loop over the models.
for model_id in models:
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)

    model_name = model_id.split("/")[-1]
    target_dir = Path(local_dir) / model_name
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir
    )
    print("Downloaded to:", target_dir)
