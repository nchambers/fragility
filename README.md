# LLM Fragility Scoring
This project contains a benchmark dataset and scoring program to measure the brittleness/fragility of LLMs when faced with input text that contains non-ASCII characters and other noisy confounding input, like spurious characters, spacing, and punctuation.


## Environment Setup

Setup your typical environment variables. Point to your local copy of CUDA, below is an example with v12.3:

```bash
export CUDA_HOME=/usr/local/cuda-12.3  
export CPATH=$CUDA_HOME/include:$CPATH  
export PATH=$CUDA_HOME/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## Mamba package Setup

You can use the environments.yml file provided to directly create a *fragility* environment:  
    ```mamba env create -f environment.yml```

Or install the packages manually from scratch yourself:

1. Create a mamba environment  
    ```mamba create -n name_env```
2. Activate your environment  
    ```mamba activate name_env```
3. Install packages  
    ```mamba install -c conda-forge -c pytorch transformers pytorch torchvision flash-attn-layer-norm spacy nltk tiktoken protobuf accelerate sentencepiece bitsandbytes datasets```


## Download Your Specific LLMs

This software is designed to test LLMs that you have already cached/downloaded on your machine. 

Your first step is to download a model and note its drive location. You can use our download tool, if you wish:

```
python tools/download_models.py /data/models/
```

This will download the models you list at the top of `download_models.py` to your /data/models/ directory. You must then look inside `models.py` and make sure you have a function that wraps your chosen LLM. Several are in there, but you'll have to make a new one for any additional models. Give it a function name, and that is the name you put in the command-line call to `fragility.py` below under "How to Run".


## How to Run

Command-line:  
```
python fragility.py [-outdir dir] [-quant int] model_name
```

Example:  
```
python fragility.py -outdir /data/outputs/ llama3b_in
```

Score the result:
```
python tools/scorer.py /data/outputs/model-llama3b_in
```


# Prompt Assistance with Chain-of-Dictionary Lookalikes

This repo also contains a program that inputs your prompt, identifies unicode lookalike characters,
and returns a string that can be added to your prompt to assist an LLM in interpreting the
text. For instance, if your prompted task is math:

    You are a helpful math assistant. Below is a basic algebra question. Please answer it:
    What is ჳ + 2?

Run the `chainofdict.py` program with the above text prompt, and it will return:

    These are hints for characters in this prompt. Use them to help answer the question:
    ჳ looks like 3

You can then add this to your prompt, such as:

    These are hints for characters in this prompt. Use them to help answer the question:
    ჳ looks like 3
    You are a helpful math assistant. Below is a basic algebra question. Please answer it:
    What is ჳ + 2?

This chain-of-dictionary approach has been shown to improve performance when your context or questions include confounding unicode characters.
