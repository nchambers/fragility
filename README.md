# LLM Fragility Scoring
This project contains a benchmark dataset and scoring program to measure the brittleness/fragility of LLMs when faced with input text that contains non-ASCII characters and other noisy confounding input, like spurious characters, spacing, and capitalization.

## Environment Setup

Setup your typical environment variables. Point to your local copy of CUDA, below is an example with v12.3:

```bash
export CUDA_HOME=/usr/local/cuda-12.3  
export CPATH=$CUDA_HOME/include:$CPATH  
export PATH=$CUDA_HOME/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## Package Setup

### Using pip

1. **Create a Python virtual environment**  
   ```python -m venv name_virtual_env```

2. **Activate your new environment**  
    ```source name_virtual_env/bin/activate```

3. **Install the dependencies**  
    ```pip install -r requirements.txt```

### Using mamba

1. **Create a mamba environment**  
    ```mamba create -n name_env```

2. **Activate your environment**  
    ```mamba activate name_env```

3. **Install packages**  
    ```mamba install -c conda-forge -c pytorch transformers pytorch torchvision flash-attn-layer-norm spacy nltk tiktoken protobuf accelerate sentencepiece bitsandbytes datasets```



## How to Run

Command-line:  
```python fragility.py [-outdir dir] [-quant int] model_name```

Example:  
```python fragility.py -outdir /data/outputs/ llama3b_in```
