#
# Main program for the Fragility Tool.
#

import models
import os
import sys
from tqdm import tqdm
import random
import torch
import phone_number
from randomizer import randomize_phone
import library
from utility import *
import utility
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class Fragility():
    def __init__(self, model, batch_size=16):
        self.outdir = "/data/outputs" # Output directory for all test output files.
        self.model = model
        self.chars = library.mk_lbry('data/chars.txt')
        self.hints = generate_hints(self.chars)
        self.batch_size = batch_size
        print(f"Fragility tool initialized with {type(model)} and batch size: {self.batch_size}")
        # random.seed(42)

    # unicode_ascii_test
    def what_is_this_unicode_baseline(self,outputFile,typee="digits",new_tokens=5):
        """
        This test will test an LLMs ability to correctly guess the unicode lookalike to its ASCII counterpart  i.e. 𝞐 looks like A.
        Will write the target, input, guess, and output.
        
        Args:
            typee (str): "digits" or "alphabet"
            outputFile (str): name of outputfile
            new_tokens (int): number of new tokens
        """

        random.seed(242)
        if typee == 'digits':
            random.seed(243)

        alphabet_prompt = """I will give you a unicode character which looks similar to an ASCII letter a-z or A-Z. Your task is to determine what ASCII letter the unicode character most likely resembles. The letter I give is case-sensitive.\nQuestion: What ASCII character is most similar to this unicode character: 𝞐\nAnswer: A\nQuestion: What ASCII character is most similar to this unicode character: 𝟋\nAnswer: f\nQuestion: What ASCII character is most similar to this unicode character:"""
        digit_prompt = """I will give you a unicode character that looks similar to a digit. Your task is to determine what digit that unicode character looks the most similar to.\nQuestion: What ASCII digit is most similar to this unicode character: 𝟚\nAnswer: 2\nQuestion: What ASCII digit is most similar to this unicode character: 𝟞\nAnswer: 6\nQuestion: What ASCII digit is most similar to this unicode character:"""
        outpath = os.path.join(self.outdir, outputFile)

        # will remove I, i, O, and o. This is because these letters are not included in the chars.txt, because they look like 1 and 0. (iso chars.txt)
        alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        # alphabet = ["A", "B"]
        digits = [ str(i) for i in range(0,10) ]
        # digits = [ str(i) for i in range(0,2) ]

        scores = []
        testing_list = alphabet if typee == "alphabet" else digits
        prompt = alphabet_prompt if typee == "alphabet" else digit_prompt
        inputs,targets,positions = [],[],[]

        for x in testing_list:
            for u in self.chars[x]:
                inputs.append(f"{prompt} {u}\nAnswer: ")
                targets.append(x)
                positions.append(u)

        with torch.no_grad():
            with open(outpath,'w') as file:
                for batch_start in tqdm(range(0,len(inputs),self.batch_size)):
                    batch = inputs[batch_start:batch_start+self.batch_size]
                    outputs = self.model.generate(batch,tokens=new_tokens,batch=True)
                    for j,output in enumerate(outputs):
                        # print(f"Checking j={j} with input str length {len(batch[j])} and output str length {len(output)}")
                        index = batch_start + j
                        input_length = len(batch[j])
                        # print(f"Target: {targets[index]}, Challenge: {positions[index]}")
                        # print(f"Generated output: {output}")
                        # print(f"Checking unicode on: {output[input_length:]}")
                        if len(output) > input_length:
                            guess = get_unicode_guess(output[input_length:])
                        else:
                            guess = ''
                        # print(f"Got guess: **{guess}**")
                        file.write(f"Target: {targets[index]}\n")
                        file.write(f"Given: {positions[index]}\n")
                        file.write(f"Guessed: {guess}\n")
                        file.write(f"Generated: {output}\n")
                        file.write("-"*50+"\n")
                        # print(f"Equal? {targets[index]==guess}")
                        # print(f"ord() {ord(targets[index])} {ord(guess) if len(guess)==1 else ""}")
                        scores.append(targets[index]==guess)
        accuracy_scores = calculate_per_target_accuracy(testing_list, zip(targets, scores))
        return accuracy_scores

    def noise_test(self,outputFile, N=5000, lookalike=True, spelling=False, leet=False, noise=True, char_prob=0.3, filler_noise=0):
        """This test adds noise to a phone number based on filler_noise."""

        # Unique seed based on parameters as long as probs are >= 0.1
        rseed = 2500 + int(1*lookalike + 2*spelling + 4*leet + 8*noise + 160*char_prob + 320*filler_noise)
        random.seed(rseed)

        prompt_phones = [ "2144357735", "7754263233", "5677381287"]
        prompt_corrupt = [ randomize_phone(x, lookalike=lookalike, spelling=spelling, leet=leet, noise=noise, char_prob=char_prob, filler_prob=filler_noise) for x in prompt_phones ]
        prompt = f"You are a phone number decoder. A phone number has been corrupted with unicode characters and punctuation. Your task is to recover just the numerical digits of that phone number. Only respond with the digits of the phone number and no other symbols. Here are a few examples.\nQ: What are the digits of this phone number: {prompt_corrupt[0]}\nA: {prompt_phones[0]}\nQ: What are the digits of this phone number: {prompt_corrupt[1]}\nA: {prompt_phones[1]}\nQ: What are the digits of this phone number: {prompt_corrupt[2]}\nA: {prompt_phones[2]}\nQ: What are the digits of this phone number:"
        outpath = os.path.join(self.outdir, outputFile)

        nums = phone_number.generate_phone("data/area_codes.txt", N)

        perturbed_dict = {}
        for num in nums:
            perturbed_dict[num] = randomize_phone(num, lookalike=lookalike, spelling=spelling, leet=leet, noise=noise, char_prob=char_prob, filler_prob=filler_noise)

        inputs,targets,maniupulated = [],[],[]
        
        for original,num in perturbed_dict.items():
            inputs.append(f"{prompt} {num}\nA: ")
            targets.append(original)
            maniupulated.append(num)
        
        correct = 0
        with torch.no_grad():
            with open(outpath,'w') as file:
                for batch_start in tqdm(range(0,len(inputs),self.batch_size)):
                    batch = inputs[batch_start:batch_start+self.batch_size]
                    outputs = model.generate(batch,tokens=15,batch=True)
                    for j, output in enumerate(outputs):
                        index = batch_start + j
                        result = get_phone_guess(output[len(batch[j]):])

                        file.write(f"Target: {targets[index]}\n")
                        file.write(f"Given: {maniupulated[index]}\n")
                        file.write(f"Guessed: {result}\n")
                        file.write(f"Generated: {output}\n")
                        file.write("-"*50+"\n")

                        if check_equality(targets[index],result):
                            correct += 1
        print(f"noise test accuracy {correct}/{N} = {correct/N}")
        accuracy_scores = correct/N
        return accuracy_scores
 
    def plain_testing_unicode_multiple(self,outputFile,cot=False,cod=False,N=5000,percent_sub=0.33):
        """
        This test will generate 30 random phone numbers and mutate the 5th digit to a lookalike ascii character.
        If Chain of Dictionary is True, it will pass a "hint" to the prompt, such as "੧ looks like 9". Else, not.
        Prompt will be the same for both apart from the "hint". Idea is to get the model to "think" through the problem.
        """

        # Unique seed based on bools
        rseed = 3400 + int(1*cot + 2*cod + 40*percent_sub)
        random.seed(rseed)

        if not cot:
            prompt =  """You are a phone number decoder. Your task is to recover the digits of a phone number that may contain unicode characters that look similar to the original digit. Only respond with the digits."""
            fewshot = """Q: What are the digits of this phone number: 9169ꬽ10801\nA: 9169010801\nQ: What are the digits of this phone number: 6613𝟳65133\nA: 6613765133\nQ: What are the digits of this phone number: 8182２18347\nA: 8182218347\nQ: What are the digits of this phone number: """
            max_tokens_to_generate = 25
        else:
            prompt = """You are a phone number decoder. Your task is to recover the digits of a phone number that may contain unicode characters that look similar to the original digit. Follow these three steps: First, break apart the given phone number into individual digits. Second, replace any lookalike unicode charcters with their ASCII digits, keeping all others the same. Third, print the ten digit phone number in ASCII only."""
            fewshot = "What are the digits of this phone number: "
            max_tokens_to_generate = 400
        outpath = os.path.join(self.outdir, outputFile)

        nums = phone_number.generate_phone("data/area_codes.txt",N)
        # print(f"Generated {len(nums)} phone numbers.")
        
        perturbed_dict = {}
        for num in nums:
            perturbed_dict[num] = replace_some_digits_with_unicode(num,self.chars,percent_sub)
        # print(f"Perturbed {len(perturbed_dict)} phone numbers with unicode characters.")
        # print(perturbed_dict)
        
        if cod:
            accuracy_scores = generate_and_evaluate_multidigit(self.model,perturbed_dict,self.hints,prompt,outpath,self.batch_size,cod,fewshot,max_tokens_to_generate)
            return accuracy_scores

        else:
            accuracy_scores = generate_and_evaluate_multidigit(self.model,perturbed_dict,None,prompt,outpath,self.batch_size,False,fewshot,max_tokens_to_generate)
            return accuracy_scores

    # This function can perform CoT and CoD through its parameters.
    def plain_testing_unicode_single(self,outputFile,cot=False,cod=False,N=30):
        """
        Unicode substitution of one perturbed digit in the middle of the phone number.
        - Default mode: few-shot 
        - cot=true: Chain of thought prompting with no few-shot examples.
        - cod=true: Add the gold answer unicode->digit in a hint. Can be used with or without cot=true
        """

        # Unique seed based on bools
        rseed = 1200 + int(1*cot + 2*cod)
        random.seed(rseed)

        if not cot:
            prompt =  """You are a phone number decoder. Your task is to recover the digits of a phone number that may contain unicode characters that look similar to the original digit. Only respond with the digits."""
            fewshot = """Q: What are the digits of this phone number: 9169ꬽ10801\nA: 9169010801\nQ: What are the digits of this phone number: 6613𝟳65133\nA: 6613765133\nQ: What are the digits of this phone number: 8182２18347\nA: 8182218347\nQ: What are the digits of this phone number:"""
            max_tokens_to_generate = 25
        else:
            prompt = """You are a phone number decoder. Your task is to recover the digits of a phone number that may contain unicode characters that look similar to standard ASCII digits. Follow these three steps: First, break apart the given phone number into individual characters. Second, replace any lookalike unicode characters with their ASCII digits, keeping all others the same. Third, print the ten digit phone number in ASCII only."""
            fewshot = "What are the digits of this phone number: "
            max_tokens_to_generate = 400
        outpath = os.path.join(self.outdir, outputFile)

        nums = phone_number.generate_phone("data/area_codes.txt", N)
        perturbed_dict = generate_all_perturbations(self.chars,nums)

        if cod:
            accuracy_scores = generate_and_evaluate_singledigit(self.model,perturbed_dict,self.hints,prompt,outpath,self.batch_size,cod,fewshot,max_tokens_to_generate)
            return accuracy_scores

        else:
            accuracy_scores = generate_and_evaluate_singledigit(self.model,perturbed_dict,None,prompt,outpath,self.batch_size,False,fewshot,max_tokens_to_generate)
            return accuracy_scores

    # CoT NOT TESTED -- DON'T KNOW HOW TO EXTRACT ITS ANSWER FROM ITS GENERATION
    def what_is_this_word(self,outputFile,cot=False,cod=False,mode='lookalike',noise_prob=0.0,N=300):
        """
        This test will generate N random words and mutate a portion of its letters to lookalike characters.
        If Chain of Dictionary is True, it will pass a "hint" to the prompt, such as "੧ looks like 9". Else, not.
        Prompt will be the same for both apart from the "hint".

        Args:
            mode (str): The mode of operation. Can be 'lookalike', 'underscore', or 'plain'.
            noise_prob (float): Probability of adding noise after each char in the string. 
            N (int): Number of unique words to include in this test.
        """
        assert mode in ('lookalike','underscore','plain'), "Mode must be one of 'lookalike', 'underscore', or 'plain'"

        # Unique seed based on bools
        rseed = 4300 + int(1*cot + 2*cod)
        random.seed(rseed)

        # Few shot learning examples of distorted words and their originals.
        exans = ( 'gained', 'choose', 'horses' )
        ex = list(exans) # plain copy
        if mode == 'lookalike':
            ex = ['g𝝰i𝖓ed','cհoos𝑒','ｈors𝔢s']
        elif mode == 'underscore':
            ex = ['g_i_ed','c_oos_','_ors_s']
        # Add noise?
        if noise_prob>0:
            ex[0] = ex[0][0]+'.'+ex[0][1:3]+'--'+ex[0][3:5]+'?'+ex[0][5]
            ex[1] = ex[1][0]+'/'+ex[1][1]+'##'+ex[1][2:5]+'.'+ex[1][5]
            ex[2] = ex[2][0]+'&&&'+ex[2][1:3]+'('+ex[2][3:6]

        # Build the prompts
        if not cot:
            prompt = """You are a word decoder. Your task is to recover the original word from a given string that contains unicode characters that look similar to standard ASCII letters. Only respond with the decoded word."""
            fewshot = f"Q: What is this word: {ex[0]}\nA: {exans[0]}\nQ: What is this word: {ex[1]}\nA: {exans[1]}\nQ: What is this word: {ex[2]}\nA: {exans[2]}\nQ: What is this word:"
            max_tokens_to_generate = 5
        else:
            prompt = """You are a word decoder. Your task is to recover the original word from a given string that contains unicode characters that look similar to standard ASCII letters. Follow these three steps: First, break apart the given word into individual characters. Second, replace any lookalike unicode characters with their ASCII letters, keeping all others the same. Third, print the word in ASCII only."""
            fewshot = "What is this word: "
            max_tokens_to_generate = 400
        outpath = os.path.join(self.outdir, outputFile)

        words = get_words(N)
        if mode == 'lookalike':
            perturbed_dict = {word: replace_some_letters_with_unicode(word,self.chars,0.33) for word in words}
        elif mode == 'underscore':
            perturbed_dict = {word: replace_some_letters_with_unicode(word,self.chars,0.33) for word in words}
            perturbed_dict = {word: utility.replace_unicode_with_underscores(pert) for word,pert in perturbed_dict.items()}
        else: # not perturbed in plain mode
            perturbed_dict = {word:word for word in words}

        # Add noise if specified
        if noise_prob > 0.0:
            perturbed_dict = {word: utility.add_noise(pert, noise_prob) for word,pert in perturbed_dict.items()}

        if cod:
            accuracy_score = generate_and_evaluate_word(self.model,perturbed_dict,self.hints,prompt,outpath,self.batch_size,cod,fewshot,max_tokens_to_generate)
            return accuracy_score
        else:
            accuracy_score = generate_and_evaluate_word(self.model,perturbed_dict,None,prompt,outpath,self.batch_size,False,fewshot,max_tokens_to_generate)
            return accuracy_score

    # CoT NOT TESTED -- DON'T KNOW HOW TO EXTRACT ITS ANSWER FROM ITS GENERATION
    def what_is_this_sentence(self,outputFile,cot=False,cod=False,unicode_prob=0.1,noise_prob=0.0,N=300):
        """
        This test will generate N random words and mutate a portion of its letters to lookalike characters.
        If Chain of Dictionary is True, it will pass a "hint" to the prompt, such as "੧ looks like 9". Else, not.
        Prompt will be the same for both apart from the "hint".

        Args:
            mode (str): The mode of operation. Can be 'lookalike', 'underscore', or 'plain'.
            noise_prob (float): Probability of adding noise after each char in the string. 
            N (int): Number of unique words to include in this test.
        """

        # Unique seed based on bools
        rseed = 4300 + int(1*cot + 2*cod)
        random.seed(rseed)

        # Few shot learning examples of distorted words and their originals.
        exans = ( 'The horse jumped over the fence.', 'Running is really good for your health!', 'I enjoy playing video games.' )
        ex = ['The հorse jump𝔢d over the f𝔢nce.', 'Run𝖓ing is re𝝰lly good for your he𝝰ltհ!', 'I 𝑒njoy playing video gam𝔢s.']
        # Add noise?
        if noise_prob>0:
            ex[0] = ex[0][0:4]+'.'+ex[0][4:13]+'--'+ex[0][13:15]+'?'+ex[0][15:]
            ex[1] = ex[1][0]+'/'+ex[1][1:7]+'##'+ex[1][7:12]+'.'+ex[1][12:]
            ex[2] = ex[2][0:6]+'&&&'+ex[2][6:9]+'('+ex[2][9:]

        # Build the prompts
        if not cot:
            prompt = """You are a sentence decoder. Your task is to recover the original sentence from a given string that contains unicode characters that look similar to standard ASCII letters. Only respond with the decoded sentence."""
            fewshot = f"Q: What is this sentence: {ex[0]}\nA: {exans[0]}\nQ: What is this sentence: {ex[1]}\nA: {exans[1]}\nQ: What is this sentence: {ex[2]}\nA: {exans[2]}\nQ: What is this sentence:"
            max_tokens_to_generate = 50
        else:
            prompt = """You are a sentence decoder. Your task is to recover the original sentence from a given string that contains unicode characters that look similar to standard ASCII letters. Follow these three steps: First, break apart the given sentence into individual characters. Second, replace any lookalike unicode characters with their ASCII letters, keeping all others the same. Third, print the sentence in ASCII only."""
            fewshot = "What is this sentence: "
            max_tokens_to_generate = 400
        outpath = os.path.join(self.outdir, outputFile)

        sentences = get_sentences(N)
        perturbed_dict = {sent: replace_some_letters_with_unicode(sent,self.chars,unicode_prob) for sent in sentences}

        # Add noise if specified
        if noise_prob > 0.0:
            perturbed_dict = {sent: utility.add_noise(pert, noise_prob) for sent,pert in perturbed_dict.items()}

        if cod:
            accuracy_score = generate_and_evaluate_sentence(self.model,perturbed_dict,self.hints,prompt,outpath,self.batch_size,cod,fewshot,max_tokens_to_generate)
            return accuracy_score
        else:
            accuracy_score = generate_and_evaluate_sentence(self.model,perturbed_dict,None,prompt,outpath,self.batch_size,False,fewshot,max_tokens_to_generate)
            return accuracy_score

def get_unicode_guess(output:str):
    output = output.strip()
    if len(output) > 0:
        return output[0]  # the first character it printed
    else:
        return ""
        
def generate_and_evaluate_singledigit(model,perturb_dict,hint_dict,prompt,output_file,batch_size=16,cod=False,fewshot='',new_tokens=25):
    """
    Given a phone number and a list of versions of it where only one middle digit is swapped out, 
    this function calls the given model to guess the original phone number.
    It compares the generations to the original phone numbers and returns the overall accuracy, as well as per-digit accuracies.

    Args:
        model (class): intialized model is passed here
        perturb_dict (dict): dictionary with list of perturbations: "4107543721" -> ["4107B43721", "4107S43721"]
        hint_dict (bool|dict): None or dictionary of unicode characters as keys and hint strings as values: "[lookalike] looks like [ASCII]"
        prompt (str): string prompt fed to tokenizer and model
        output_file (str): name of the output file to write results
        batch_size (int): size of the batching (16 default)
        cod (bool): boolean for if chain of dictionary is active
        fewshot (str): few-shot examples to include
    """
    frag_scores = [[] for _ in range(10)]
    inputs,targets,postions,manipulated = [],[],[],[]

    for original,pn in perturb_dict.items():
        for num in pn:
            # If chain-of-dictionary, add a hint.
            hint = ''
            if cod:
                hint = f"\nHere is a hint: {hint_dict[num[4]]}."

            inputs.append(f"{prompt}{hint}\n{fewshot} {num}\nA: ")
            targets.append(original)
            postions.append(int(original[4]))
            manipulated.append(num)

    with torch.no_grad():
        with open(output_file,'w') as file:
            for batch_start in tqdm(range(0,len(inputs),batch_size)):
                batch = inputs[batch_start:batch_start+batch_size]
                outputs = model.generate(batch,new_tokens,batch=True)
                for j, output in enumerate(outputs):
                    index = batch_start + j
                    result = get_phone_guess(output[len(batch[j]):])

                    file.write(f"Target: {targets[index]}\n")
                    file.write(f"Given: {manipulated[index]}\n")
                    file.write(f"Guessed: {result}\n")
                    file.write(f"Generated: {output}\n")
                    file.write("-"*50+"\n")

                    frag_scores[postions[index]].append(check_equality(targets[index],result))
    accuracy_scores = calculate_accuracy(frag_scores)
    return accuracy_scores

# NOTE: Very similar to above generate_and_evaluate_singledigit() function: perturb_dict is different and we don't track specific digit accuracies
def generate_and_evaluate_multidigit(model,perturb_dict,hint_dict,prompt,output_file,batch_size=16,cod=False,fewshot='',new_tokens=25):
    """
    Given a phone number and one perturbed version of it, this function calls the given model to guess the original phone number.
    It compares the generations to the original phone numbers and returns the overall accuracy.

    Args:
        model (class): intialized model is passed here
        perturb_dict (dict): dictionary with perturbations: "4107543721 -> 4Io754B72i"
        hint_dict (bool|dict): None or dictionary of unicode characters as keys and hint strings as values: "[lookalike] looks like [ASCII]"
        prompt (str): string prompt fed to tokenizer and model
        output_file (str): name of the output file to write results
        batch_size (int): size of the batching (16 default)
        cod (bool): boolean for if chain of dictionary is active
        fewshot (str): few-shot examples to include

    Returns:
       accuracy_score (float): accuracy over the given perturbations
    """
    inputs,targets,manipulated = [],[],[]

    for original,pn in perturb_dict.items():
        hints = ''

        # If chain-of-dictionary, add hints.
        if cod:
            hints = f"\nHere are some hints: "
            for digit in pn:
                if digit.isascii() == False:
                    hints += hint_dict[digit] + '. '

        inputs.append(f"{prompt}{hints}\n{fewshot} {pn}\nA: ")
        targets.append(original)
        manipulated.append(pn)

    correct = 0
    with torch.no_grad():
        with open(output_file,'w') as file:
            for batch_start in tqdm(range(0,len(inputs),batch_size)):
                batch = inputs[batch_start:batch_start+batch_size]
                outputs = model.generate(batch,new_tokens,batch=True)
                for j, output in enumerate(outputs):
                    index = batch_start + j
                    result = get_phone_guess(output[len(batch[j]):])

                    file.write(f"Target: {targets[index]}\n")
                    file.write(f"Given: {manipulated[index]}\n")
                    file.write(f"Guessed: {result}\n")
                    file.write(f"Generated: {output}\n")
                    file.write("-"*50+"\n")

                    if check_equality(targets[index],result):
                        correct += 1

    accuracy_score = correct/len(targets)
    return accuracy_score

def generate_and_evaluate_word(model,perturb_dict,hint_dict,prompt,output_file,batch_size=16,cod=False,fewshot='',new_tokens=25):
    """
    Given a word and one perturbed version of it, this function calls the given model to guess the original word.
    It compares the generations to the original words and returns the overall accuracy.

    Args:
        model (class): intialized model is passed here
        perturb_dict (dict): dictionary with perturbations: "4107543721 -> 4Io754B72i"
        hint_dict (bool|dict): None or dictionary of unicode characters as keys and hint strings as values: "[lookalike] looks like [ASCII]"
        prompt (str): string prompt fed to tokenizer and model
        output_file (str): name of the output file to write results
        batch_size (int): size of the batching (16 default)
        cod (bool): boolean for if chain of dictionary is active
        fewshot (str): few-shot examples to include

    Returns:
       accuracy_score (float): accuracy over the given perturbations
    """
    inputs,targets,manipulated = [],[],[]

    for original,pn in perturb_dict.items():
        hints = ''

        # If chain-of-dictionary, add hints.
        if cod:
            hints = f"\nHere are some hints: "
            for digit in pn:
                if digit.isascii() == False:
                    hints += hint_dict[digit] + '. '

        inputs.append(f"{prompt}{hints}\n{fewshot} {pn}\nA:")
        targets.append(original)
        manipulated.append(pn)

    correct = 0
    with torch.no_grad():
        with open(output_file,'w') as file:
            for batch_start in tqdm(range(0,len(inputs),batch_size)):
                batch = inputs[batch_start:batch_start+batch_size]
                outputs = model.generate(batch,new_tokens,batch=True)
                for j, output in enumerate(outputs):
                    index = batch_start + j
                    result = utility.get_word_guess(output[len(batch[j]):])

                    file.write(f"Target: {targets[index]}\n")
                    file.write(f"Given: {manipulated[index]}\n")
                    file.write(f"Guessed: {result}\n")
                    file.write(f"Generated: {output}\n")
                    file.write("-"*50+"\n")

                    if utility.check_letter_equality(targets[index],result):
                        correct += 1

    accuracy_score = correct/len(targets)
    return accuracy_score

def generate_and_evaluate_sentence(model,perturb_dict,hint_dict,prompt,output_file,batch_size=16,cod=False,fewshot='',new_tokens=25):
    """
    Given a word and one perturbed version of it, this function calls the given model to guess the original word.
    It compares the generations to the original words and returns the overall accuracy.

    Args:
        model (class): intialized model is passed here
        perturb_dict (dict): dictionary with perturbations: "4107543721 -> 4Io754B72i"
        hint_dict (bool|dict): None or dictionary of unicode characters as keys and hint strings as values: "[lookalike] looks like [ASCII]"
        prompt (str): string prompt fed to tokenizer and model
        output_file (str): name of the output file to write results
        batch_size (int): size of the batching (16 default)
        cod (bool): boolean for if chain of dictionary is active
        fewshot (str): few-shot examples to include

    Returns:
       accuracy_score (float): accuracy over the given perturbations
    """
    inputs,targets,manipulated = [],[],[]

    for original,pn in perturb_dict.items():
        hints = ''

        # If chain-of-dictionary, add hints.
        if cod:
            hints = f"\nHere are some hints: "
            for digit in pn:
                if digit.isascii() == False:
                    hints += hint_dict[digit] + '. '

        inputs.append(f"{prompt}{hints}\n{fewshot} {pn}\nA:")
        targets.append(original)
        manipulated.append(pn)

    correct = 0
    with torch.no_grad():
        with open(output_file,'w') as file:
            for batch_start in tqdm(range(0,len(inputs),batch_size)):
                batch = inputs[batch_start:batch_start+batch_size]
                outputs = model.generate(batch,new_tokens,batch=True)
                for j, output in enumerate(outputs):
                    index = batch_start + j
                    result = utility.get_sentence_guess(output[len(batch[j]):])

                    file.write(f"Target: {targets[index]}\n")
                    file.write(f"Given: {manipulated[index]}\n")
                    file.write(f"Guessed: {result}\n")
                    file.write(f"Generated: {output}\n")
                    file.write("-"*50+"\n")

                    result = result[0:len(targets[index])]  # truncate to target length for sentence comparison
                    if utility.check_sentence_equality(targets[index],result):
                        correct += 1

    accuracy_score = correct/len(targets)
    return accuracy_score

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python fragility.py [options] <model_name>")
        print("Example: python fragility.py -outdir /tmp/outputs -quant 4 deepseek_14b")
        sys.exit(1)

    # Parse the args
    outdir = None
    if "-outdir" in sys.argv:
        outdir_index = sys.argv.index("-outdir")
        if outdir_index + 1 < len(sys.argv):
            outdir = sys.argv[outdir_index + 1]

    qn = None
    if "-quant" in sys.argv:
        quant_index = sys.argv.index("-quant")
        if quant_index + 1 < len(sys.argv):
            qn = int(sys.argv[quant_index + 1])

    model_name = sys.argv[-1]

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

    # Compute batch size based on model type
    if model_name in ['phi2','llama1b','llama1b_in','llama3b','llama3b_in','vicuna7b','deepseek_8b','deepseek_14b','mistral_small','mistral_small_in']:
        batch_size = 32
    else:
        batch_size = 16

    # Initialize the Fragility tool with the model
    tool = Fragility(model,batch_size=batch_size)
    # Create output subdir just for this model
    if outdir is not None:
        tool.outdir = outdir
    tool.outdir = tool.outdir + f"/model-{model_name}/"
    os.makedirs(tool.outdir, exist_ok=True)
    # How many base phone numbers to use. Used ~4 for variation, but results similar to N=1.
    N = 1

    overall_grade = 0
    grades = {}

    print(f"Testing Model {type(model)}")

    # WHAT IS THIS WORD?
    print('What is this word unicode test:')
    accuracy = tool.what_is_this_word("whatisthis-word.txt",cot=False,cod=False,N=300,mode='lookalike')
    grades['whatisword-unicode'] = accuracy
    print(f"Overall: {accuracy}")
    print('What is this word underscore test:')
    accuracy = tool.what_is_this_word("whatisthis-word-under.txt",cot=False,cod=False,N=300,mode='underscore')
    grades['whatisword-underscore'] = accuracy
    print(f"Overall: {accuracy}")
    accuracies = []
    for i in range(1,6):
        print(f'What is this word with noise test {i}:')
        acc = tool.what_is_this_word(f"whatisthis-word-noise{i}.txt",cot=False,cod=False,N=300,mode='plain',noise_prob=i/5)
        accuracies.append( round(acc,3) )
        print(f"Overall: {acc}")
    grades['whatisword'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], accuracies)] )
    print(f"Grade: {grades[list(grades)[-1]]} Noise accuracies: {accuracies}")
  
    # NOT NEEDED
    # print('What is this word with unicode AND noise test:')
    # accuracy = tool.what_is_this_word("whatisthis-word-uni-and-noise.txt",cot=False,cod=False,N=300,mode='lookalike',noise_prob=2/5)
    # grades['whatisworduni'] = accuracy
    # print(f"Overall: {accuracy}")

    print('What is this sentence test:')
    accuracy = tool.what_is_this_sentence("whatisthis-sent.txt",cot=False,cod=False,unicode_prob=0.1,N=300)
    grades['whatissent'] = accuracy
    print(f"Overall: {accuracy}")

    # Single similarity question: B -> 8
    print('What is this letter unicode test:')
    accuracies = tool.what_is_this_unicode_baseline("whatisthis-let.txt", typee="alphabet")
    grades['whatislet'] = sum(accuracies)/len(accuracies)
    print(f"Overall: {round(grades[list(grades)[-1]],3)} Individual: {accuracies}")

    # Single similarity question: 8 -> 8
    print('What is this digit unicode test:')
    accuracies = tool.what_is_this_unicode_baseline("whatisthis-dig.txt", typee="digits")
    grades['whatisdig'] = sum(accuracies)/len(accuracies)
    print(f"Overall: {round(grades[list(grades)[-1]],3)} Individual: {accuracies}")

    # One middle digit: 410 -> 4I0
    print('Middle digit test:')
    accuracies = tool.plain_testing_unicode_single("plainphone.txt",N=N,cod=False)  # N is number of base phone numbers, perturbed with all possible unicode
    grades['middle'] = sum(accuracies)/10
    print(f"Overall: {round(grades[list(grades)[-1]],3)} Individual: {accuracies}")

    print('Middle digit CoD test:')
    accuracies = tool.plain_testing_unicode_single("plainphone-CoD.txt",N=N,cod=True)
    grades['midCoD'] = sum(accuracies)/10
    print(f"Overall: {round(grades[list(grades)[-1]],3)} Individual: {accuracies}")

    print('Middle digit CoT test:')
    accuracies = tool.plain_testing_unicode_single("plainphone-CoT.txt",N=N,cot=True,cod=False)
    grades['midCoT'] = sum(accuracies)/10
    print(f"Overall: {round(grades[list(grades)[-1]],3)} Individual: {accuracies}")

    print('Middle digit CoT and CoD test:')
    accuracies = tool.plain_testing_unicode_single("plainphone-CoT-CoD.txt",N=N,cot=True,cod=True)
    grades['midCoTCoD'] = sum(accuracies)/10
    print(f"Overall: {round(grades[list(grades)[-1]],3)} Individual: {accuracies}")

    print('Multiple digit CoT test:')
    accuracy = tool.plain_testing_unicode_multiple("plainphone-multi-CoT.txt",N=200,cot=True,cod=False)
    grades['multiCoT'] = accuracy
    print(f"Overall: {accuracy}")

    print('Multiple digit CoT and CoD test:')
    accuracy = tool.plain_testing_unicode_multiple("plainphone-multi-CoT-CoD.txt",N=200,cot=True,cod=True)
    grades['multiCoTCoD'] = accuracy
    print(f"Overall: {accuracy}")

    # Noise: 410 -> 4..1*0
    print('Noise test:')
    accuracies = []
    for i in range(1,6):
        acc = tool.noise_test(f"noise{i}.txt", N=500, lookalike=False, spelling=False, leet=False, noise=True, char_prob=i/5, filler_noise=i/5)
        accuracies.append( round(acc,3) )
        # print(f"{round(i/5,2)}: {accuracies[-1]}")
    grades['noise'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], accuracies)] )
    print(f"Grade: {grades[list(grades)[-1]]} Noise accuracies: {accuracies}")
    
    # Noise: 410 -> 4oneZeRO
    print('Spelling test:')
    accuracies = []
    for i in range(1,6):
        acc = tool.noise_test(f"noise-spell{i}.txt", N=500, lookalike=False, spelling=True, leet=False, noise=False, char_prob=i/5, filler_noise=i/5)
        accuracies.append( round(acc,3) )
    grades['spell'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], accuracies)] )
    print(f"Grade: {grades[list(grades)[-1]]} Spelling accuracies: {accuracies}")

    # Noise: 410 -> 4oneZeRO
    print('Lookalike + Noise test:')
    accuracies = []
    for i in range(1,6):
        acc = tool.noise_test(f"noise-lookalike-noise{i}.txt", N=500, lookalike=True, spelling=False, leet=False, noise=True, char_prob=i/5, filler_noise=0.33)
        accuracies.append( round(acc,3) )
    grades['lookalike_noise'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], accuracies)] )
    print(f"Grade: {grades[list(grades)[-1]]} Lookalike + Noise accuracies: {accuracies}")

    # TRANSFERRED MMLU CODE TO THE NEW benchmarks.py FROM chain.py. 
    # -- potentially remove chain.py completely after this.

    from benchmarks import Benchmarks
    B = Benchmarks(model, outdir=tool.outdir)

    # MMLU norm
    print('MMLU test pure:')
    grades['mmlu'] = B.mmlu_test("mmlu.txt", seed=142, altered_prob=0.0)
    print(f"Accuracy: {round(grades[list(grades)[-1]],3)}")
    
    # MMLU 20% noise
    print('MMLU test with 20% perturbed')
    grades['mmlu2'] = B.mmlu_test("mmlu-20.txt", seed=143, altered_prob=0.2)
    print(f"Accuracy: {round(grades[list(grades)[-1]],3)}")


    # Grade for all phone/digit tests.
    phone_grade = 0.2*grades['whatisdig'] + \
        0.1*grades['middle'] + 0.1*grades['midCoD'] + 0.1*grades['midCoT'] + 0.1*grades['midCoTCoD'] + \
        0.1*grades['multiCoTCoD'] + \
        0.1*grades['noise'] + 0.1*grades['spell'] + 0.1*grades['lookalike_noise']

    # Grade for phone/digit but no CoD.
    phone_noCoD_grade = 0.2*grades['whatisdig'] + \
        0.1*grades['middle'] + 0.2*grades['midCoT'] + \
        0.2*grades['multiCoT'] + \
        0.1*grades['noise'] + 0.1*grades['spell'] + 0.1*grades['lookalike_noise']

    # Need to put MMLU and one other NLP test here.
    mmlu_grade = min(1.0, grades['mmlu2']/grades['mmlu'])
    NLU_grade = 0.5*mmlu_grade + 0.15*grades['whatisword-unicode'] + 0.15*grades['whatisword-underscore'] + 0.2*grades['whatissent']
    overall_grade = 0.3*NLU_grade + 0.1*grades['whatislet'] + 0.6*phone_grade

    print(f"PHONE grade for {model_name}: {round(phone_grade,3)}")
    print(f"PHONE grade (no CoD) for {model_name}: {round(phone_noCoD_grade,3)}")
    print(f"MMLU grade for {model_name}: {round(mmlu_grade,3)}")
    print(f"NLU grade for {model_name}: {round(NLU_grade,3)}")
    print(f"OVERALL grade for {model_name}: {round(overall_grade,3)}")
