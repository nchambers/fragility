"""
Program that generates additional prompting text for your prompt, using chain-of-dictionary (CoD) prompting. 
This is a technique for helping LLMs understand that certain unicode characters in the prompt are lookalikes 
of ASCII characters, which can be important for correctly interpreting prompts that include those characters.

Command-line use:
    python chainofdict.py "What is ჳ + 2?"
Output:
    "ჳ looks like 3"

API use:
    Create a ChainOfDict object and call create_prompt() with your prompt string.
"""

import library

class ChainOfDict:
    """Build and store the unicode-to-hint dictionary used by CoD prompting."""

    def __init__(self):
        """Load unicode lookalikes and precompute hint strings."""
        self.chars = library.mk_lbry('data/chars.txt') # dictionary: ASCII -> [ lookalikes]
        self.dict = self._generate_hints(self.chars)

    def _generate_hints(self, lookalikes):
        """
        Gernerates the dictionary that will be used for chain of dict.

        Args:
            lookalikes (dict): a dictionary mapping ASCII characters to lists of unicode lookalikes.

        Returns:
            dictionary mapping unicode lookalikes (such as ੧) to hint strings of the form "੧ looks like 9"
        """
        chain_d = {}
        for ascii_char in lookalikes:
            for similar in lookalikes[ascii_char]:
                chain_d[similar] = f"{similar} looks like {ascii_char}"
        return chain_d

    def create_prompt(self, prompt, noascii=True):
        """
        Generate hint text that can be added to your prompt, based on the unicode characters 
        found in the given prompt argument. Returns string text of the hints.

        Args:
            prompt (str): User prompt that may contain unicode lookalikes.
            noascii (bool): If True, skip ASCII characters in the hints

        Returns:
            str: A formatted hint block, or an empty string if no hints are needed.
        """

        hints = ''
        for digit in prompt:
            # Most prompts are ASCII-heavy; skipping ASCII keeps hints concise.
            if digit.isascii() and noascii:
                continue
            if digit in self.dict:
                hints += self.dict[digit] + '\n'

        if len(hints) > 0:
            hints = "These are hints for characters in this prompt. Use them to help answer the question:\n" + hints.strip()
        
        return hints




if __name__ == "__main__":
    import argparse

    # CLI utility for quickly previewing CoD hints for a prompt string.
    parser = argparse.ArgumentParser(description="Create additional prompt instructions for a prompt with unicode characters.")
    parser.add_argument("prompt", type=str, help="The prompt to an LLM.")
    args = parser.parse_args()

    chain = ChainOfDict()
    add = chain.create_prompt(args.prompt)
    print(add)