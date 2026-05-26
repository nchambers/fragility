#
# library.py
# peter dejonge, with owen pitchford, advised by dr. nate chambers
#
import random

def mk_lbry(file="data/chars.txt") -> dict:
    """
    purpose: create a dictionary/library which when calling a specific character,
    such as "A" or "0", will return a lookalike character
    input: the file which you are trying to do this process on
    """
    # will return this dictionary
    char_dict = {}

    # open this file, skip the first 9 lines (junk in the file)
    with open(file, 'r') as fh:
        for _ in range(9):
            next(fh)

        # each line will be stripped, and then it will try to set the key as the first character
        # in the line. the rest of the line then becomes a list which will serve as the value for the key.
        # reason try is because errors occur without it.
        # return the dictionary
        for line in fh:
            line = line.strip()
            try:      
                key = line[0]
                value = list(line[1:])

                if key in char_dict:
                    char_dict[key].append(value)
                else:
                    char_dict[key] = value
            except:
                pass
    return char_dict

def num_dict() -> dict:
    """
    purpose: create a dictionary mapping numbers to its counterpart spelled out (9 -> nine)
    input: none
    output: dictionary
    """
    n_d = {}
    with open('data/num.txt', 'r') as fh:
        for line in fh:
            line = line.strip().split(',')
            n_d[line[0]] = line[1]
    return n_d

def leet_dict() -> dict:
    """
    purpose: creates a dictionary which maps letters to their leet counterpart
    input: none
    returns: dictionary
    """
    l_d = {}
    with open('data/leet.txt', 'r') as fh:
        for line in fh:
            line = line.strip().split(',')
            l_d[line[0]] = line[1:]
    return l_d

def leet_translator(string) -> str:
    """
    purpose: translate a string into simple leet text
    input: string
    output: string
    """
    ld = leet_dict()
    res = []
    for i in string:
        if i in ld:
            res.append( random.choice(ld[i]) )
        else:
            res.append(i)
    return ''.join(map(str,res))

def digit_to_word(digit):
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return digit_words[digit]



if __name__ == "__main__":
    questions, choices, answers = mmlu_dataset()

    print(questions[:5])
    print(choices[:5])
    print(answers[:5])