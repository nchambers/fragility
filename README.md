# LLM Fragility Scoring
This project contains a benchmark dataset and scoring program to measure the brittleness/fragility of LLMs when faced with input text that contains non-ASCII characters and other noisy confounding input, like spurious characters, spacing, and capitalization.

## Setup

Setting up your Environment to run the LLMs

======= Example Environment =======
export CUDA_HOME=/usr/local/cuda-12.3
export CPATH=$CUDA_HOME/include:$CPATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH


======= Package Setup =======
-- Using pip
Step 1:
    Create a python virtual environment 
    ~$ python -m name_virtual_env

Step 2:
    Activate your new environment
    ~$ source name_virtual_env/bin/activate

Step 3:
    Install the dependencies
    ~$ pip install -r requirements.txt

-- Using mamba
Step 1:
    Create a mamba environment
    ~$ mamba create -n name_env

Step 2:
    Activate your environment
    ~$ mamba actiavte name_env

Step 3:
    Install packages (might not need spacy+nltk, could test this later)
    ~$ mamba install -c conda-forge -c pytorch transformers pytorch torchvision flash-attn-layer-norm spacy nltk tiktoken protobuf accelerate sentencepiece bitsandbytes datasets



## How to Run

Command-line:
python fragility.py [-outdir <dir>] [-quant <int>] <model_name>

Example:
python fragility.py -outdir /data/outputs/ llama3b_in
