import re
import random
import os
from randomizer import noise_symbols

def get_words(N):
    """
    Reads the words from data/wordlist_6.txt and returns them in a list.
    Args:
        N (int): Number of words to return.
    Returns:
        list: A list of N random words.
    """

    with open('data/wordlist_6.txt', 'r') as file:
        words = file.read().splitlines()
    
    if N > len(words):
        raise ValueError(f"Requested {N} words, but only {len(words)} are available.")
    
    # Remove few-shot examples
    words = [word for word in words if word not in ["gained", "choose", "horses"]]

    return random.sample(words, N)

def get_sentences(N):
    """
    Samples sentences from the MMLU dataset's questions and returns them in a list.
    Args:
        N (int): Number of sentences to return.
    Returns:
        list: A list of N strings (sentences)
    """

    # Get the sentences from the MMLU questions
    from benchmarks import mmlu_dataset
    questions,_,_ = mmlu_dataset(random_sample=N*5)

    # Loop over question text, and grab the first sentence from each. Put in a new sentences variable.
    sentences = [q.split(".")[0] for q in questions]

    # Choose sentences with medium length. Only alphanumeric. No "Which" questions (so there aren't lots of the same).
    sentences = [s for s in sentences if (60 < len(s) < 140) and re.match(r'^[A-Za-z0-9\s.,!?\'"]+$', s) and not s.startswith("Which")]
    if N > len(sentences):
        raise ValueError(f"Requested {N} sentences, but only {len(sentences)} are available.")

    return sentences[0:N]

def get_phone_guess(generated):
    # trim off the leading prompt
    # res = res[len(prompt[i]):]
    matches = re.findall(r'\d{10}|\d{3}-\d{3}-\d{4}', generated)
    if len(matches) > 0:
        return matches[-1].replace("-","")
    else:
        return "Incomplete Guess"

def get_word_guess(generated):
    generated = generated.strip().lower()

    if generated.startswith("answer"):
        generated = generated.split("answer")[-1].strip()
    if generated.startswith("a:"):
        generated = generated.split("a:")[-1].strip()

    # If blank string
    if len(generated) == 0:
        return ""

    # Take the first word if there are multiple words
    generated = generated.split()[0]

    return generated

def get_sentence_guess(generated):
    generated = generated.strip().lower()

    if generated.startswith("answer:"):
        generated = generated[len("answer:"):].strip()
    if generated.startswith("a:"):
        generated = generated[len("a:"):].strip()

    return generated.strip()


def open_for_writing(filepath):
    """
    Returns an open file handle to the given filepath for writing.
    If the file already exists, append a number to the end of the filename. For example, "file.txt" becomes "file2.txt", then "file3.txt", and so on.

    Args:
        filepath (str): The original desired file path.

    Returns:
        tuple: A pair containing:
            - file_handle (file object): An open file handle for writing.
            - new_filepath (str): The actual file path that was opened.
    """
    directory, filename = os.path.split(filepath)
    base_name, extension = os.path.splitext(filename)

    counter = 1
    new_filepath = filepath

    # Loop until a non-existent file path is found
    while os.path.exists(new_filepath):
        counter += 1
        new_filename = f"{base_name}{counter}{extension}"
        new_filepath = os.path.join(directory, new_filename)

    file_handle = open(new_filepath, 'w')
    return file_handle, new_filepath

def random_case(word):
    return ''.join(random.choice([char.upper(), char.lower()]) for char in word)

def digit_to_word(digit):
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return random_case(digit_words[digit])

def replace_some_digits_with_words(phone_number, prob=0.333):
    assert prob <= 1 and prob >= 0, "Probability must be between 0 and 1."

    new_number_parts = []
    for d in phone_number:
        if random.random() < prob:
            new_number_parts.append(digit_to_word(int(d)))
        else:
            new_number_parts.append(d)
    return "".join(new_number_parts)

def replace_some_digits_with_unicode(phone_number, unicode, prob=0.333):
    """ Given one string phone number, return one string phone number with some digits replaced with unicode. """
    assert prob <= 1 and prob >= 0, "Probability must be between 0 and 1."

    new_number_parts = []
    for d in phone_number:
        if random.random() < prob and str(d) in unicode:
            new_number_parts.append(random.choice(unicode.get(str(d))))
        else:
            new_number_parts.append(d)
    return "".join(new_number_parts)

def add_noise(text, prob=0.0, rep=0.3) -> str:
    """ 
    Given a string, return the same string with punctuation/whitespace filler between characters.
    Setting prob=0 returns the original string.
    
    Args:
        text (str): The input string to randomize.
        prob (float): Probability of inserting noise after a character.
        rep (float): Probability of repeating the noise character again, inserting multiple chars after a character.

    Returns:
        str: The string with noise added
    """

    assert prob <= 1 and prob >= 0, "Probability must be between 0 and 1."

    # How many locations to add filler?
    num_to_add = int(len(text) * prob + 0.5)
    # Randomly select which indices to add filler after
    indices = list(range(len(text)))
    indices_to_add = random.sample(indices, num_to_add)

    nu = []
    for i, d in enumerate(text):
        nu.append(d)
        if i in indices_to_add:
            noisechar = random.choice(noise_symbols)
            while True:
                nu.append(noisechar)
                if random.random() > rep: 
                    break
    return "".join(nu)


def replace_letters_and_add_noise(text, unicode, prob=0.333, filler_prob=0.0, filler_rep=0.3) -> str:
    """ 
    Given a string, return the same string but with some letters replaced with unicode.
    Also (optionally) adds punctuation/whitespace filler between characters.
    Setting both char_prob=0 and filler_prob=0 returns the original string.
    
    Args:
        text (str): The input string to randomize.
        unicode (dict): Precomputed dict from char->list of unicode
        char_prob (float): Probability of swapping an ASCII character out for a lookalike.
        filler_prob (float): Probability of inserting noise after a character.
        filler_rep (float): Probability of repeating the noise character again, inserting multiple chars after a character.

    Returns:
        str: The randomized string.
    """

    # First do the unicode swaps
    text = replace_some_letters_with_unicode(text,unicode,prob)
    # Add noise characters
    return add_noise(text, filler_prob, filler_rep)


def replace_some_letters_with_unicode(word, unicode, prob=0.333):
    """ 
    Given one string word, return one string word with some letters replaced with unicode.
    Precomputes how many chars to replace based on given 'prob', then randomly chooses that many.
    """
    assert prob <= 1 and prob >= 0, "Probability must be between 0 and 1."

    num_to_replace = int(len(word) * prob + 0.5)
    
    # Remove indices that don't have unicode lookalikes
    indices = list(range(len(word)))
    goodindices = []
    for i in indices:
        if str(word[i]) in unicode:
            goodindices.append(i)
    # Randomly select indices to replace
    indices_to_replace = random.sample(goodindices, num_to_replace)

    # Replace the selected indices with random lookalikes
    new_word_parts = []
    for i, d in enumerate(word):
        if i in indices_to_replace:
            # new_word_parts.append(random.choice(unicode.get(str(d))))
            new_word_parts.append(random.choice(unicode[d]))
        else:
            new_word_parts.append(d)
    return "".join(new_word_parts)
    
def replace_unicode_with_underscores(word):
    """ Given one string word, return it with unicode characters replaced with underscores. """
    new_word_parts = []
    for d in word:
        if not d.isascii():
            new_word_parts.append('_')
        else:
            new_word_parts.append(d)
    return "".join(new_word_parts)
   
def generate_perturbations_space(lookalikes,numbers):
    """
    Generates a dictionary of key phone numbers, and their corresponding lists of perturbations
    Key: Mutated phone number (5th digit will be changed from 0-9)
    Value: List which contains all perturbations of that phone number for that index
    """
    perturbed_dict = {}
    
    for num in numbers:
        for i in range(10):
            mutated_num = num[:8]+str(i)+num[9:]
            perturbed = []
            for replacement in lookalikes[str(i)]:
                perturbed.append(mutated_num[:8]+replacement+mutated_num[9:])
            perturbed_dict[mutated_num] = perturbed

    return(perturbed_dict)
def generate_perturbations_n2w(lookalikes,numbers):
    """
    Generates a dictionary of key phone numbers, and their corresponding lists of perturbations
    Key: Mutated phone number (5th digit will be changed from 0-9)
    Value: List which contains all perturbations of that phone number for that index
    """
    perturbed_dict = {}
    
    for num in numbers:
        for i in range(10):
            mutated_num = num[:4]+str(i)+num[5:]
            perturbed = []
            perturbed.append(mutated_num[:4]+random_case(lookalikes[i])+mutated_num[5:])
            perturbed_dict[mutated_num] = perturbed

    return(perturbed_dict)

def reverse_lookup(dictionary, value):
    # Loop through the dictionary items and check for matching values
    for key, val in dictionary.items():
        if value in val:
            return key
    return None  # Return None if value is not found

def check_equality(expected_output, actual_output):
    cleaned_output = re.sub(r'\D', '', actual_output)
    expected_output = re.sub(r'\D', '', expected_output)
    return expected_output == cleaned_output

def check_letter_equality(expected_output, actual_output):
    cleaned = re.sub(r"[^A-Za-z]",'',actual_output)
    gold = re.sub(r"[^A-Za-z]",'',expected_output)
    return cleaned == gold

def check_sentence_equality(expected_output, actual_output):
    """ True if all letters match ignoring case, spaces, and non-letters. """
    cleaned = re.sub(r"[^A-Za-z]",'',actual_output.lower())
    gold = re.sub(r"[^A-Za-z]",'',expected_output.lower())
    return cleaned == gold

def calculate_per_target_accuracy(targets, fragility_bools):
    '''
    Calculates accuracy for each character's tests.

    Args:
    fragility_bools (list of pairs): (target, bool_correct)
    '''
    correct_counts = [0]*len(targets)
    total_counts = [0]*len(targets)
    for target, bool_correct in fragility_bools:
        if bool_correct:
            correct_counts[targets.index(target)] += 1
        total_counts[targets.index(target)] += 1
    return [round(correct/total,3) for correct, total in zip(correct_counts, total_counts)]

def calculate_accuracy(fragility_scores):
    '''
    Given multiple lists of scores, calculates the accuracy for each list.

    Args:
        fragility_scores (list of lists): Each inner list contains boolean values e.g., (True,False,True,True)

    Returns:
        list: A list of accuracy scores for each inner list
    '''
    accuracy_scores = []
    for scores in fragility_scores:
        correct = sum(scores)
        total = len(scores)
        accuracy = correct/total
        accuracy_scores.append(round(accuracy,3))
    
    # sum all true values in all inner lists of fragility_scores
    total_correct = sum(sum(scores) for scores in fragility_scores)
    total_length = sum(len(scores) for scores in fragility_scores)
    print(f"Accuracy = {total_correct/total_length} from Total Correct: {total_correct}, Total Length: {total_length}")

    return accuracy_scores

def new_calculate_accuracy(real_scores,random_scores):
    total_len = len(real_scores) + len(random_scores)
    real_accuracy = sum(real_scores) / total_len
    random_accuracy = (sum(real_scores) + sum(random_scores)) / total_len
    return real_accuracy,random_accuracy

def generate_all_perturbations(lookalikes,numbers):
    """
    Given a list of phone numbers, create corresponding lists of perturbed versions.
    Return a dictionary:
       Key: phone number 
       Value: List which contains all perturbations of that phone number (5th digit is changed)
    """
    perturbed_dict = {}
    
    for num in numbers:
        for i in range(10):
            mutated_num = num[:4]+str(i)+num[5:]
            perturbed = []
            for replacement in lookalikes[str(i)]:
                perturbed.append(mutated_num[:4]+replacement+mutated_num[5:])
            perturbed_dict[mutated_num] = perturbed

    return perturbed_dict

def generate_hints(lookalikes):
    """
    Gernerates the dictionary that will be used for chain of dict
    Key: the lookalike ascii character (such as ੧)
    Value: "੧ looks like 9"
    """
    chain_d = {}

    for i in range(10):
        for similar in lookalikes[str(i)]:
            chain_d[similar] = f"{similar} looks like {str(i)}"
    return chain_d

# def generate_perturbations(perturbation_type_dict, numbers, N):
#     '''
#     Generates a dictionary for a list of phone numbers.
#     Key: Original Phone Number
#     Value: Nested list of size 10. Each index contains all perturbions of that
#     phone number for that index
#     '''
#     perturbed_dict = dict.fromkeys(numbers)
#     for num in numbers:
#         temp = []
#         for i in range(10):
#             if(N == 1): 
#                 new_num = num
#             elif(N == 2):
#                 new_num = get_one_perturb(perturbation_type_dict, num, 4, None)
            
#             temp.append(get_all_perturbs(perturbation_type_dict, new_num, 5, i))
#         perturbed_dict[num] = temp 
    
#     return perturbed_dict

# def get_all_perturbs(perturbation_type_dict, phone_number, index, digit):
#     """
#     Perturb a phone number based on the specified type and index to apply the perturbation to.
#     Returns a list of numbers off all unicode substitutions for digit given.

#     Parameters:
#     - perturbation_type_dict (dict): A map of digits to perturbation.
#     - phone_number (str): str of phone number as 555-555-5555
#     - index (int): Location to perturb 0-9.
#     - digit (int): What number to substitute.
#     """
#     if digit == None:
#         digit = random.randint(0,9)

#     num = list(phone_number)
#     perturbed_numbers = []

#     for val in perturbation_type_dict[str(digit)]:
#         num[index] = val
#         perturbed_numbers.append(''.join(num))

#     return perturbed_numbers

# def get_one_perturb(perturbation_type_dict, phone_number, index, digit):
#     if digit == None:
#         digit = random.randint(0,9)
    
#     num = list(phone_number)
#     num[index] = random.choice(perturbation_type_dict[str(digit)])
#     return ''.join(num)

def format_phone_number(number):
    '''
    Returns a string format of a a 10 digit number in format 555-555-5555
    '''
    if(type(number) != str): number = str(number)

    return f'{number[:3]}-{number[3:6]}-{number[6:]}'


def print_percent_done(index, total, bar_len=50, title='Please wait'):
    '''
    index is expected to be 0 based index.
    0 <= index < total
    '''
    percent_done = (index)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}:\t[{done_str}{togo_str}] {int(percent_done)}% done', end='\r')

    if round(percent_done) == 100:
        print(f'\t✅{title}:\t[{done_str}{togo_str}] 100% done',end="\r")



if __name__ == "__main__":
    import phone_number
    import library
    # f = generate_all_perturbations('data/chars.txt',phone_number.generate_phone('data/area_codes.txt',30))
    f = generate_hints(library.mk_lbry('data/chars.txt'))
    print(f)

    uni = library.mk_lbry('data/chars.txt')
    x = replace_some_digits_with_unicode("4109328765", uni, prob=.33)
    print(x)




