#
# Run this once to create a dataset of words of a specific length.
#
# Download: https://github.com/first20hours/google-10000-english
# curl https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/google-10000-english.txt
#

# Number of desired words
N = 500
# Length of words that you want
LENGTH = 6


import random

# Read file of English words, one per line
try:
    with open('google-10000-english.txt') as f:
        common_words = f.read().splitlines()
except FileNotFoundError:
    print("File not found. Please make sure the file exists. Use curl with the URL in this program's comments.")
    exit(1)

# Filter words to get only those of length LENGTH
filtered_words = [word for word in common_words if len(word) == LENGTH]

# Randomly select N words from the filtered list
random.sample(filtered_words, N)

# Save the selected words to a file
with open(f'wordlist_{LENGTH}.txt', 'w') as f:
    for word in random.sample(filtered_words, N):
        f.write(word + '\n')    