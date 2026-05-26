#
# Wrapper class for OpenAI models.
#
# You need an API key for a paid OpenAI account set in your environment:
# export OPENAI_API_KEY="your_api_key_here"
#
# For the OpenAI library:
# mamba install conda-forge::openai 
#
import os
import sys
import random
fragility_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fragility"))
sys.path.append(fragility_dir)

import openai
import asyncio
import backoff
from LLM import LLM


# Check for API key
if not 'OPENAI_API_KEY' in os.environ:
    print("You must set the environment variable OPENAI_API_KEY.")
    print("The OpenAI class requires an API key to a paid OpenAI account.")
    exit(1)
    

class OpenAIModel(LLM):

    def __init__(self, model='gpt-4o-mini', temperature=1):
        ''' @model Specify a valid OpenAI model name as a string '''
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.client = openai.AsyncOpenAI()
        self.syncclient = openai.OpenAI()
        
    def make_single_api_call_to_gpt(self, prompt, tokens=20, batch=True):
        response = self.syncclient.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    async def make_api_call_to_gpt(self, prompt, tokens=20, batch=True):
        retry_delay = 1  # initial delay in seconds
        max_retries = 5  # maximum number of retries

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except openai.APIConnectionError as e:
                print(f"API connection error on attempt {attempt + 1}: {e}")
            except openai.RateLimitError as e:
                print(f"Rate limit hit on attempt {attempt + 1}: {e}")
            except openai.APIStatusError as e:
                print(f"Server returned an error on attempt {attempt + 1}: {e}")

            # Wait before retrying, with jitter
            await asyncio.sleep(retry_delay + random.uniform(0, 1))
            retry_delay *= 2  # exponential backoff

        raise RuntimeError(f"Failed after {max_retries} attempts")


        # response = await self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0,
        # )
        # return response.choices[0].message.content

    async def make_calls(self, prompts):
        # Asynchronously call the function for each prompt
        tasks = [self.make_api_call_to_gpt(pro) for pro in prompts]
        self.results = await asyncio.gather(*tasks)

    def generate(self, prompt, tokens=20, batch=True):
        ''' This method takes in a List of string prompts and returns a list of generations '''

        # Convert to list if given a single str.
        if isinstance(prompt, str):
            prompt = [prompt]

        # Call with 50 prompts at a time.
        generated = []
        chunk_size = 50
        for start in range(0,len(prompt),chunk_size):
            subprompts = prompt[start:start+chunk_size]
            print(f"GPT Processing: {len(subprompts)} prompts...")

            if( len(subprompts) == 1 ):
                self.results = [ self.make_single_api_call_to_gpt(subprompts[0]) ]
            else:
                asyncio.run(self.make_calls(subprompts))

            print("*****************")
            print(self.results)
            fullgens = [ pro+' '+gen for pro,gen in zip(subprompts,self.results) ]
            generated.extend(fullgens)
        
        print('ALL DONE!')

        if not batch:
            return generated[0]
        else:
            return generated

    def cleanup(self):
        ''' Nothing to do for an API '''
        pass

if __name__ == '__main__':
    print('Hi!')

    #openai = OpenAIModel(model='gpt-3.5-turbo-0125')
    openai = OpenAIModel(model='gpt-4o-mini')

    prompt = "You are an expert phone number decoder. Please identify the original phone number in the following, and only tell me the number, no explanation. What is this phone number: 306-8𝟜9-3783"

    prompts = ["You are an expert phone number decoder. Please identify the original phone number in the following, and only tell me the number, no explanation. What is this phone number: 306-8𝟜9-3783", \
               "You are an expert phone number decoder. Please identify the original phone number in the following, and only tell me the number, no explanation. What is this phone number: 306-8𑢬9-3783", \
                "You are an expert phone number decoder. Please identify the original phone number in the following, and only tell me the number, no explanation. What is this phone number: 306-8𑢬9-3783" \
                ]

    prompts = [prompt]*10

    gen = openai.generate(prompts)
    
    print('Generation:', gen)
