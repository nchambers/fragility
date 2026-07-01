#
# models.py
# python3 models.py
# peter dejonge, with owen pitchford, advised by dr. nate chambers
#
# mamba install -c conda-forge -c pytorch transformers pytorch torchvision flash-attn-layer-norm spacy nltk tiktoken protobuf accelerate sentencepiece
#
import torch
import re
import sys
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from LLM import LLM
from OpenAI import OpenAIModel

def optimal_gpu():
    """
    Return a string (i.e., "cuda:1") of the GPU that has the most free memory at the moment.
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        return "cpu"
    gpu_mem = []
    for i in range(n_gpu):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        gpu_mem.append((i,free_mem))
    
    b_gpu = max(gpu_mem, key=lambda x: x[1])[0]
    return f"cuda:{b_gpu}"

def cleanup(model):
    """ Remove from GPU and delete """
    # Delete the model object
    del model
    # Force garbage collection
    import gc
    gc.collect()
    # Clear CUDA cache
    torch.cuda.empty_cache()

class BatchModel(LLM):
    def __init__(self,filepath,device,quantization=None):
        self.pth = filepath
        self.dv = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.pth)
        if quantization == 8:
            qn_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == 4:
            qn_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
        else:
            qn_config = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pth,
            quantization_config=qn_config,
            torch_dtype=torch.bfloat16 if quantization is None else None,
            attn_implementation="flash_attention_2" if quantization in (8,None) else None,
            device_map="balanced"
            #device_map="auto"
        )
        self.model = torch.compile(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def generate(self,prompt,tokens=20,batch=True,general_output=True):
        inputs = self.tokenizer(prompt,padding=True,return_tensors='pt')

        outputs = self.model.generate(input_ids=inputs['input_ids'].to(self.dv),
                                    attention_mask=inputs['attention_mask'].to(self.dv),
                                    max_new_tokens=tokens,
                                    use_cache=True,
                                    do_sample=False)
                                    # temperature=1e-6)
        if batch:
            # decode the output and return it
            results = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
            return results
        else:
            # decode the output and return it
            result = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
            return result
        
    def cleanup(self):
        cleanup(self.model)
        return


class GenericModel(LLM):
    """
    Basics of running the models. Will take a filepath and device to initialize the model on the GPU. Then calling generate will allow for the input of a prompt.
    """
    def __init__(self,filepath,device,quantization=None):  
        """
        Args:
            filepath (str): The path to the model directory.
            device (str): The device to run the model on, e.g., "cuda:0" or "cpu". Use "auto" for multiple GPUs.
            quantization (int, optional): The quantization level, either 8 or 4 bits. Defaults to None.
        """
        self.pth = filepath
        print(f'Generic Model loading from {self.pth} on {device}')
        #self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pth)
        # optional quantization
        qn_config = None
        if quantization == 8:
            qn_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == 4:
            qn_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
       
        # load the model to the device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pth,
            quantization_config=qn_config,
            torch_dtype=torch.bfloat16 if qn_config is None else None,
            attn_implementation="flash_attention_2" if quantization in (8,None) else None,
            device_map="auto" # ="cuda" uses whichever CUDA_VISIBLE_DEVICES=1 you set on command line
        ) #.to(self.device) # don't need this, the device_map will handle it
        self.model = torch.compile(self.model)
        self.device = next(self.model.parameters()).device  # get the device of the first layer for calling generate()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def generate(self,prompt,tokens=20,batch=True):
        inputs = self.tokenizer(prompt,padding=True,return_tensors='pt')

        outputs = self.model.generate(input_ids=inputs['input_ids'].to(self.device),
                                    attention_mask=inputs['attention_mask'].to(self.device),
                                    max_new_tokens=tokens,
                                    use_cache=True,
                                    do_sample=False)
                                    # temperature=1e-6)
        if batch:
            # decode the output and return it
            results = self.tokenizer.batch_decode(outputs,skip_special_tokens=True)
            return results
        else:
            result = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
            return result

    def cleanup(self):
        cleanup(self.model)
        return


class phi2(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/Phi2",device=optimal_gpu(),quantization=qn)

class phi3_4b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/phi3-mini-4b",device=optimal_gpu(),quantization=qn)

class phi3_7b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/phi3-small-7b-instant",device=optimal_gpu(),quantization=qn)

class phi3_14b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/phi3-med-14b",device=optimal_gpu(),quantization=qn)

class gemma1b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/gemma-3-1b-it",device=optimal_gpu(),quantization=qn)

class gemma4b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/gemma-3-4b-it",device=optimal_gpu(),quantization=qn)

class gemma12b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/gemma-3-12b-it",device=optimal_gpu(),quantization=qn)

class gemma27b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/gemma-3-27b-it",device=optimal_gpu(),quantization=qn)

class llama1b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/Llama3.2-1B",device=optimal_gpu(),quantization=qn)

class llama1b_in(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Llama3.2-1B-Instruct/",device=optimal_gpu(),quantization=qn)
    
class llama3b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Llama3.2-3B/",device=optimal_gpu(),quantization=qn)
    
class llama3b_in(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Llama3.2-3B-Instruct/",device=optimal_gpu(),quantization=qn)

class llama70b(BatchModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Llama3.3-70B/",device=optimal_gpu(),quantization=qn)

class llama70b_in(BatchModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Llama3.3-70B-Instruct/",device=optimal_gpu(),quantization=qn)
        
class qwen0p6b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/qwen3-0.6b",device=optimal_gpu(),quantization=qn)

class qwen1p7b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/qwen3-1.7b",device=optimal_gpu(),quantization=qn)

class qwen4b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/qwen3-4b",device=optimal_gpu(),quantization=qn)

class qwen8b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/qwen3-8b",device=optimal_gpu(),quantization=qn)

class qwen14b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/qwen3-14b",device=optimal_gpu(),quantization=qn)

class qwen32b(GenericModel):
    def __init__(self,qn=None):     
        super().__init__("/home/scs/nchamber/data/models/qwen3-32b",device=optimal_gpu(),quantization=qn)

class vicuna7b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/vicuna-7B/",device=optimal_gpu(),quantization=qn)

class vicuna13b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/vicuna-13B/",device=optimal_gpu(),quantization=qn)

class mixtral7b(BatchModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Mixtral-7B/",device=optimal_gpu(),quantization=qn)

class mistral_small(GenericModel): # 24B
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/Mistral-Small/",device=optimal_gpu(),quantization=qn)

class mistral_small_in(GenericModel): # 24B
    def __init__(self, qn=None):
        super().__init__("/home/scs/nchamber/data/models/Mistral-Small-Instruct/",device=optimal_gpu(),quantization=qn)

class deepseek_1p5b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/DeepSeek-R1-Distill-Qwen-1.5B/",device=optimal_gpu(),quantization=qn)

class deepseek_7b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/DeepSeek-R1-Distill-Qwen-7B/",device=optimal_gpu(),quantization=qn)

class deepseek_14b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/DeepSeek-R1-Distill-Qwen-14B/",device=optimal_gpu(),quantization=qn)

class deepseek_32b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/DeepSeek-R1-Distill-Qwen-32B/",device=optimal_gpu(),quantization=qn)

class olmo1b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/OLMo-2-0425-1B-Instruct/",device=optimal_gpu(),quantization=qn)

class olmo7b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/OLMo-2-1124-7B-Instruct/",device=optimal_gpu(),quantization=qn)

class olmo13b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/OLMo-2-1124-13B-Instruct/",device=optimal_gpu(),quantization=qn)

class olmo32b(GenericModel):
    def __init__(self,qn=None):
        super().__init__("/home/scs/nchamber/data/models/OLMo-2-0325-32B-Instruct/",device=optimal_gpu(),quantization=qn)

class gpt_3_5_turbo(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-3.5-turbo-0125')
        self.temperature = 0

class gpt_o4_mini(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('o4-mini-2025-04-16')

class gpt_4o(OpenAIModel): # most expensive
    def __init__(self, qn=None):
        super().__init__('gpt-4o-2024-11-20')
        self.temperature = 0

class gpt_4o_mini(OpenAIModel):
    def __init__(self, qn=None):
        #super().__init__('gpt-4o-mini-2024-07-18')
        super().__init__('gpt-4o-mini')

class gpt_4_1_mini(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-4.1-mini')

class gpt_4_1(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-4.1')

class gpt_5_mini(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-5-mini')

class gpt_5_4_mini(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-5.4-mini')

class gpt_5_4(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-5.4')

class gpt_5_5(OpenAIModel):
    def __init__(self, qn=None):
        super().__init__('gpt-5.5')

class tester(GenericModel):
    """
    Testing model that just returns "tester" for every generation.
    """
    def __init__(self,qn=None):
        pass

    def generate(self,prompt,tokens=20,batch=True):
        return ['tester'] * len(prompt)
            
    def cleanup(self):
        pass

if __name__ == '__main__':
    """
    How to run models.py
    1. Ensure you have the proper environment to run it. Look at "README_env.txt" to ensure this is complete.
    2. All models already have the type of model which runs it most efficiently (BatchModel or GenericModel), for large or small.
        Therefore, to run it, all you need to do is give it arguments.

        For GenericModels (Llama1B, Llama1B_Instruct, Llama3B, Llama3B_Instruct, Vicuna7B, Vicuna13B, DeepSeek8B, and DeepSeek14B):
            model = llama1b_in()
            result = model.generate(prompt (list or str),
                                    tokens=20 (int),
                                    batch=True (bool),
                                    general_output=False (bool)
                                    )
            print(result)

        For BatchModels (Llama70B, Mixtral7b, Mistral_Small, Mistral_Small_Instruct, DeepSeek32B, DeepSeek70B):
            model = llama70b(qn=None (bool or int))
            result = model.generate(prompt (list or str),
                                    tokens=20 (int),
                                    batch=True (bool),
                                    general_output=False (bool)
                                    )
            print(result)
    """

    # print('Optimal gpu:', optimal_gpu())
    # oneb = mistral_small_in()
    # print(oneb.generate(input('prompt'),600,False,False))
    #oneb = llama70b()
    oneb = phi2()
    #print(oneb.model)
    f = oneb.generate(input('enter here: '))
    print(f)
