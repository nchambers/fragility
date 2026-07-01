# 
# This program analyzes the tokenization of unicode lookalikes in various models. 
# It checks if the unicode characters are present in the tokenizer, compares their embeddings to their ASCII counterparts, and computes rankings based on cosine similarity.
#
# python token_checker.py <model_name>
#
import sys
import string
import phone_number
import torch
import library
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class TokenChecker():
    def __init__(self,model):
        self.model = model
        self.model_name = type(self.model).__name__
        self.chars = library.mk_lbry('data/chars.txt')

    def mmlu_token_sizes(self):
        """ Computes the tokenization size of MMLU multiple-choice questions, with and without unicode lookalikes. """
        # Get 1000 MMLU choices (the A-D letter options).
        from randomizer import Randomize
        from benchmarks import mmlu_dataset
        questions,choices,answers = mmlu_dataset(random_sample=1000)
        choices_with_uni = Randomize(prob=0.2, filler_prob=0).batch_randomize(choices)

        # Loop over choices and get token lengths after tokenization
        num_tokens,num_tokens_uni = 0,0
        for i in range(len(choices)):
            for j in range(len(choices[i])):
                choice = choices[i][j]
                choice_uni = choices_with_uni[i][j]
                tokens = self.model.tokenizer.tokenize(choice)
                tokens_uni = self.model.tokenizer.tokenize(choice_uni)
                num_tokens += len(tokens)
                num_tokens_uni += len(tokens_uni)
                # print(f"Original: {choice}\tTokens: {tokens}\tLength: {len(tokens)}")
                # print(f"Altered: {choice_uni}\tTokens: {tokens_uni}\tLength: {len(tokens_uni)}")

        print(f"Average tokens per set of choices:\t {num_tokens/len(choices):.2f}")
        print(f"Average tokens per set with unicode:\t {num_tokens_uni/len(choices):.2f}")

    def get_token_id(self, tokenizer, uni):
        prefix_token_id = tokenizer.convert_tokens_to_ids(['▁'])[0] if len(tokenizer.convert_tokens_to_ids(['▁'])) > 0 else -1

         # Encode the string.
        ids = tokenizer.encode(uni)
        # Remove BOS and EOS IDs if they are in there
        while ids[0] == tokenizer.bos_token_id or ids[0] == prefix_token_id:
            ids = ids[1:]
        while ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
        # If only one token ID from encoding, then the unicode char is in the vocab
        if len(ids) == 1:
            return ids[0]
        else:
            return None
        
    def get_embedding(self, model, tokenizer, uni):
        """ Helper function: get the embedding for a unicode character if it exists in the tokenizer """
        # Get the token ID for this unicode character
        token_id = self.get_token_id(tokenizer, uni)
        if token_id == None:
            return None
        
        inputs = tokenizer(uni, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0") # put on GPU with the model
        # Preferred HuggingFace API (works for Gemma)
        if hasattr(model.model, "get_input_embeddings"):
            embedding_layer = model.model.get_input_embeddings()
        # Backward-compatible fallback (works for Vicuna, Mistral, Mixtral)
        elif hasattr(model.model, "embed_tokens"):
            embedding_layer = model.model.embed_tokens

        token_embeddings = embedding_layer(input_ids) # get the embeddings for these tokens
        i = torch.where(input_ids[0] == token_id)[0] # find the embedding index for our unicode char
        embed = token_embeddings[0][i]
        return embed

    def reverse_uni_lookup(self,uni):
        """ Helper function: lookup an ASCII that has the given unicode in its lookalikes list """
        for ch in string.digits+string.ascii_letters:
            looks = self.chars.get(ch,[])
            if uni in looks:
                return ch

    def get_all_unicode_chars(self):
        """ Helper function: get all unicode lookalikes without repeats """
        allunis = []
        digitunis = []
        seen = set()
        for ch in string.digits+string.ascii_letters:
            for uni in self.chars.get(ch,[]):
                # Skip if uni is an ASCII character (some are in the list of lookalikes)
                if uni in string.digits+string.ascii_letters:
                    continue
                # if not seen before
                if uni not in seen:
                    seen.add(uni)
                    allunis.append(uni)
                # If it's a digit lookalike, add to digit list too
                if ch in string.digits:
                    digitunis.append(uni)
        return allunis,digitunis

    def compare_embeddings(self):
        """
        Compare the embeddings of unicode lookalikes to their original ASCII characters.
        Compute the average ranking of the correct ASCII in its vector similarities to all ASCII.
        """
        tk = self.model.tokenizer

        # Some models (Vicuna, Mixtral) have a prefix token when seeing unique character sets
        prefix_token_id = tk.convert_tokens_to_ids(['▁'])[0] if len(tk.convert_tokens_to_ids(['▁'])) > 0 else -1
        # print('Prefix token: ', prefix_token_id)
        
        embeds = {}
        allunis,digitunis = self.get_all_unicode_chars()
        for uni in allunis+list(string.digits+string.ascii_letters):
            # Find the token ID (if it exists) for this unicode character
            token_id = self.get_token_id(tk, uni)
            if token_id != None:
                # print(f"Found {uni} with {token_id}")
                embed = self.get_embedding(self.model.model, tk, uni)
                # print(embed)
                embeds[uni] = embed

        # Now compare the embeddings!
        rankings,top1,top3,top5,N = 0,0,0,0,0
        ranks = []
        for k in embeds.keys():
            # Not an ASCII char (must be Unicode)
            if not k in string.digits+string.ascii_letters:
                # Compare against all ASCII characters
                scored = []
                for j in string.digits+string.ascii_letters:
                    sim = torch.nn.functional.cosine_similarity(embeds[k], embeds[j], dim=1)
                    scored.append((sim.item(),j))
                # Print top 3 matches.
                scored.sort(reverse=True)
                # print(f"{k}: {scored[0:3]}")
                # Find ranking
                ascii = self.reverse_uni_lookup(k)
                chars = [x[1] for x in scored]
                rank = chars.index(ascii)
                ranks.append(rank)
                rankings += rank
                if rank == 0:
                    top1 += 1
                if rank < 3:
                    top3 += 1
                if rank < 5:
                    top5 += 1
                N += 1
                print(f"{k}\t{ascii}\t{chars[0:5]}\trank {rank}")
        print(f"Average ranking: {rankings/N:.3f}")
        print(f"Median ranking: {sorted(ranks)[len(ranks)//2]}")
        print(f"Top 1: {top1/N:.3f}")
        print(f"Top 3: {top3/N:.3f}")
        print(f"Top 5: {top5/N:.3f}")
        print(f"Top 1 (w/unks): {top1/(len(allunis)):.3f}")
        print(f"Top 3 (w/unks): {top3/(len(allunis)):.3f}")
        print(f"Top 5 (w/unks): {top5/(len(allunis)):.3f}")

    def tokenizer_test(self):
        """
        Searches all unicode lookalikes to see if they are tokens in the tokenizer.
        Returns two percentages: % known of all letters/digits and % known of just digits
        """
        print(f"Running tokenizer_test on {self.model_name}")
        tk = self.model.tokenizer

        # Some models (Vicuna, Mixtral) have a prefix token when seeing unique character sets
        prefix_token_id = tk.convert_tokens_to_ids(['▁'])[0] if len(tk.convert_tokens_to_ids(['▁'])) > 0 else -1
        print('Prefix token: ', prefix_token_id)

        # Get all unicode lookalikes without repeats
        allunis,digit_unis = self.get_all_unicode_chars()
        print(f"Total unicode lookalikes:\t{len(allunis)}")
        print(f"Total digit unicode lookalikes:\t{len(digit_unis)}")

        # Loop over unicode
        N,known = 0,0
        Nd,knownd = 0,0
        for uni in allunis:
            # if the unicode char exists in the tokenizer
            if self.get_token_id(tk, uni) != None:
                known += 1
                if uni in digit_unis:
                    knownd += 1
            if uni in digit_unis:
                Nd += 1
            N += 1
        print(f"Tokenizer contains:\t{known}")
        print(f"...of which tokenizer contains digit lookalikes:\t{knownd}")
        
        return (known/N, knownd/Nd)

    
def testing_vocab():
    """ 
    Just looks up the unicode characters (A-Za-z0-9) in each model's tokenizer.
    Prints the percentage of unicode characters found in each tokenizer.
    """
    import models
    mods = ['llama1b', 'olmo1b', 'vicuna7b', 'deepseek_1p5b', 'gemma1b', 'qwen0p6b', 'mistral_small']

    for mod in mods:
        print(f"Initializing {mod}")
        # Load the model
        model = getattr(models,mod)()
        # Run the tests
        tool = TokenChecker(model)
        tool.mmlu_token_sizes()
        tool.tokenizer_test()
        # Embeddings sim
        tool.compare_embeddings()
        # Remove the model from memory and the GPU
        model.cleanup()

if __name__ == "__main__":

    # Command-line argument: model name
    if len(sys.argv) < 2:
        print("Usage: python token_checker.py <model_name>")
        print("Example: python token_checker.py llama3b")
        sys.exit(1)
    model_name = sys.argv[-1]

    if model_name == "all":
        testing_vocab()
        sys.exit(1)

    # Load the model by name "llama3b"
    import models
    if not hasattr(models, model_name):
        print(f"Error: Model '{model_name}' not found in models module.")
        sys.exit(1)
    model_func = getattr(models, model_name)
    model = model_func()

    # model = models.llama3b()
    # model = models.deepseek_14b()
    tool = TokenChecker(model)

    tool.mmlu_token_sizes()
    tool.tokenizer_test()
    tool.compare_embeddings()
