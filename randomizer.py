import random
from tqdm import tqdm
from library import mk_lbry, num_dict, leet_translator
noise_symbols = [
    '-', '/', '.', ',', '@', '#', '*', '_', '+', '(', ')', '[', ']', 
    '{', '}', '|', ':', ';', "'", '"', '~', '^', '&', '%', '$', '=', 
    '<', '>', '?', '!', ' '
]
digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
charlib = mk_lbry()


#
# Randomizes multiple choice text.
# Given a list of list of choices, it perturbs the choice text with characters with lookalikes and noise.
#
class Randomize():
    def __init__(self,prob=0.2,filler_prob=0.2):
        self.chars = mk_lbry()
        self.prob = prob
        self.filler_prob = filler_prob

    def batch_randomize(self,batch:list)->list:
        new_B = []
        for PROMPT in tqdm(batch):
            new_P = []
            for CHOICE in PROMPT:
                new_C = []
                for LETTER in CHOICE:
                    # Should we alter this character?
                    if (random.random() <= self.prob) and (LETTER in self.chars):
                        new_C.append(random.choice(self.chars[LETTER]))
                    else:
                        new_C.append(LETTER)
                    # Should we add noise after this character?
                    if (random.random() <= self.filler_prob):
                        new_C.append(random.choice(noise_symbols))                        
                new_P.append(''.join(map(str,new_C)))
            new_B.append(new_P)
        return new_B


def randomize_str(text,char_prob=0.3,filler_prob=0.0,filler_rep=0.3) -> str:
    """
    Randomizes a given string by substituting characters with lookalikes and adds filler noise.
    Setting both char_prob=0 and filler_prob=0 returns the original string.
    
    Args:
        text (str): The input string to randomize.
        char_prob (float): Probability of swapping an ASCII character out for a lookalike.
        filler_prob (float): Probability of inserting noise after a character.
        filler_rep (float): Probability of repeating the noise character again, inserting multiple chars after a character.

    Returns:
        str: The randomized string.
    """

    nu = []
    for ch in text:
        # Should we replace this char?
        if char_prob > 0 and random.random() <= char_prob and ch in charlib:
            lookalike = random.choice(charlib[ch])
            nu.append(lookalike)
        # Else, keep the original
        else:
            nu.append(ch)      
        # Should we add filler noise?
        if filler_prob > 0 and (random.random() <= filler_prob):
            noisechar = random.choice(noise_symbols)
            while True:
                nu.append(noisechar)
                if random.random() > filler_rep: 
                    break
    
    return ''.join(nu)


def randomize_phone(number,lookalike=True,spelling=False,leet=False,noise=False,char_prob=0.3,filler_prob=0.3,filler_rep=0.3) -> str:
    """
    Alter a given phone number based by substituting lookalike characters, leet speak, or both.

    Args:    
        phone number (str): string of ASCII digits
        lookalike (bool): True if we substitute unicode lookalikes
        leet (bool): True if we substitute leet-speak words
        noise (bool): True if we insert random punctuation noise characters between digits
        char_prob (float): Probability of swapping an ASCII digit out for a lookalike
        filler_prob (float): Probability of inserting noise after a character
        filler_rep (float): Probability of repeating the noise character again, inserting multiple chars after a digit

    Returns:
        string: The original number but with altered digits and spacings
    """

    # Setup the types of substitutions that are turned on.
    subtypes = []
    if lookalike: subtypes.append("lookalike")
    if spelling: subtypes.append("spelling")
    if leet: subtypes.append("leet")

    # The new phone number in list form.
    nu = []
    # according to booleans
    if lookalike or spelling or leet:
        # dictionary mapping characters to lookalike
        chars = mk_lbry('data/chars.txt')
        # dictionary mapping numbers to full-spelling
        spellings = num_dict()

        for i in number:

            # if the random is less than the probability
            if random.random() <= char_prob:
                subtype = random.choice(subtypes)

                # Digit to unicode lookalike.   
                if subtype=="lookalike":
                    # choose randomly from the dictionary a lookalike character
                    voo = random.choice(chars[i])
                # Digit to leet speak.
                elif subtype=="leet":
                    # translate the digit into leet i.e. 9 -> n1n3
                    voo = leet_translator(i)
                    # Capitalize it?
                    if( random.random() < 0.2 ):
                        voo = voo.upper()
                # Digit to spelling.
                else:
                    voo = digit_words[int(i)]
                    # 50/50 chance to be capitalized
                    voo = ''.join(random.choice([char.upper(), char.lower()]) for char in voo)

                nu.append(voo)

                # Add filler noise.
                if noise and (random.random() <= filler_prob):
                    voo = random.choice(noise_symbols)
                    while True:
                        nu.append(voo)
                        # Break loop if random big enough
                        if( random.random() > filler_rep ): break

            else:
                nu.append(i)
        # return the joined list is a string
        return ''.join(map(str,nu))
    
    # not lookalike nor leet
    else:
        nu = []
        for i in number:
            nu.append(i)
            if noise and (random.random() <= filler_prob):
                voo = random.choice(noise_symbols)
                while True:
                    nu.append(voo)
                    # Break loop if random big enough
                    if( random.random() > filler_rep ): break
            else:
                pass
        return ''.join(nu)
    

if __name__ == "__main__":

    r = Randomize(0.25,0)

    num = '5677381287'
    # rn = randomize(num,False,False,True, 1, 0.2)
    for _ in range(10):
        rn = randomize_phone(num, lookalike=False, spelling=True, leet=False, noise=False, char_prob=0.7, filler_prob=0.5)
        print(num,rn)

    strs = [ 'Do you want to be a pepper too?', 'The raven flies under dark skies.' ]
    for s in strs:
        print(s, '-->', randomize_str(s, char_prob=0.4, filler_prob=0, filler_rep=0.3))