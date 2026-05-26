import torch
from tqdm import tqdm
from randomizer import Randomize
from utility import *
import os
import random

class Benchmarks:
    def __init__(self,model,outdir="/data/outputs"):
        self.outdir = outdir
        self.model = model
        print(f"Benchmarks initialized with output directory: {self.outdir}")

    def mmlu_test(self,outputFile,batch_size=16,new_tokens=3,altered_prob=0.0,seed=42):
        """
        Asks the challenge questions from the MMLU benchmark.
        Optionally, the multiple-choice answers can be permuted randomly with unicode and adversarial fillers.

        Args:
            outputFile (str): The name of the output file to write the results.
            batch_size (int): The number of questions to process in each batch.
            new_tokens (int): The number of tokens to allow the model to generate. RECOMMEND: at least 3
            altered_prob (float): The probability that any character in multiple-choice answers is replaced.

        Returns:
            float: Overall accuracy on the MMLU questions.
        """
        random.seed(seed)

        # Downsample to save time. Need ~2400 for +/- 2% standard error and 95% confidence. (or 9k for +/- 1%)
        # Randomly sample 2.4k questions from the test portion.
        questions,choices,answers = mmlu_dataset(random_sample=2400)

        cs = ['A','B','C','D']
        print(f'MMLU testing {len(questions)} questions.')

        if altered_prob > 0:
            choices = Randomize(prob=altered_prob, filler_prob=0).batch_randomize(choices)

        real_scores = []
        inputs = []
        givens = []
        for i,Q in enumerate(questions):
            PROMPT = f"""Below is a multiple-choice question. Answer the question with just the letter of the correct answer.\nQuestion: {Q}\n"""
            for g,C in enumerate(choices[i]):
                PROMPT += f"{cs[g]}. {C}\n"
            PROMPT += "Answer: "
            givens.append(f"{Q} " + " ".join([f"{cs[g]}. {C}" for g,C in enumerate(choices[i])]))
            inputs.append(PROMPT)
        
        outpath = os.path.join(self.outdir, outputFile)
        # print(f"Writing MMLU test results to {outpath}")

        with torch.no_grad():
            with open(outpath,'w') as file:
                for batch_start in tqdm(range(0,len(inputs),batch_size)):
                    batch = inputs[batch_start:batch_start+batch_size]
                    outputs = self.model.generate(batch,new_tokens,True)
                    guesses = mmlu_guesses(outputs)
                    for j,(GUESS,OUTPUT,INDICATOR) in enumerate(guesses):
                        index = batch_start+j
                        file.write(f"Target: {cs[answers[index]]}\n")
                        file.write(f"Given: {givens[index]}\n")
                        file.write(f"Guessed: {GUESS}\n")
                        file.write(f"Generated: {OUTPUT[len(batch[j]):]}\n")
                        file.write("-"*50+"\n")
                        real_scores.append(check_mmlu_equality(answers[index],GUESS)) # if INDICATOR else random_scores.append(check_mmlu_equality(answers[index],GUESS))

        return real_scores.count(True) / len(real_scores)
        
def mmlu_guesses(outputs:list):
    """ Extracts the A-D guesses from model outputs. """
    guesses = []
    for out in outputs:
        found = re.findall(r"Answer:\s*([A-Da-d])",out)
        if len(found) > 0:
            guesses.append((found[0].upper(),out,True))
        else:
            #guesses.append((random.choice(['A','B','C','D']),out,False))
            guesses.append(('',out,False))
    return guesses

def check_mmlu_equality(expected_output,actual_output):
    if actual_output == "A":
        return 0 == expected_output
    elif actual_output == "B":
        return 1 == expected_output
    elif actual_output == "C":
        return 2 == expected_output
    elif actual_output == "D":
        return 3 == expected_output
    else:
        return False
    
def mmlu_dataset(datasplit='test', random_sample=None):
    """
    Args:
        datasplit: test|dev|validation|auxiliary_train
        random_sample: If specified, randomly samples this many questions from the dataset.
        seed: Random seed for reproducibility.
    """
    from datasets import load_dataset
    dataset = load_dataset("cais/mmlu",'all')
    dataset = dataset[datasplit]

    questions = []
    choices = []
    answers = []

    # Downsample the dataset if requested
    if random_sample is not None:
        rng = random.Random(42)  # Fixed seed for dataset reproducibility
        indices = rng.sample(range(len(dataset)), random_sample)
        dataset = dataset.select(indices)

    print(f"Downsampled dataset to {len(dataset)} questions.")

    # Loop over the Dataset object and fill in the three lists with questions/choices/answers
    for i in range(len(dataset)):
        questions.append(dataset[i]['question'])
        choices.append(dataset[i]['choices'])
        answers.append(dataset[i]['answer'])

    return questions,choices,answers



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benchmarks.py <model_name> [4|8]")
        print("Example: python benchmarks.py deepseek_14b 8")
        sys.exit(1)

    # Parse the args
    model_name = sys.argv[1]
    qn = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Get the model function from the `models` module
    import models
    if not hasattr(models, model_name):
        print(f"Error: Model '{model_name}' not found in models module.")
        sys.exit(1)
    model_func = getattr(models, model_name)

    # Call the model function with optional quantization
    if qn is not None:
        model = model_func(qn=qn)
    else:
        model = model_func()
    print(f"Loaded model: {model_name}")

    B = Benchmarks(model,outdir=".")
    print("Running MMLU test...")
    accuracy = B.mmlu_test("mmlu-testing.txt", new_tokens=3)
    print(f"MMLU test accuracy: {accuracy}")