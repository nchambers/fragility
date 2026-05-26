import os


def calculate_accuracy(file_path: str) -> float:
    """
    Opens a given file path, processes it to compare "Target" and "Guessed" values,
    and computes an overall accuracy score.

    The file is expected to have lines like:
    Target: some_value
    Guessed: some_predicted_value

    The function compares "Target" with "Guessed" and computes the accuracy.

    Args:
        file_path (str): The path to the text file.

    Returns:
        float: The accuracy score
    """
    correct_predictions = 0
    total_predictions = 0
    current_target = None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Strip leading/trailing whitespace from the line for robust parsing
            stripped_line = line.strip()

            if stripped_line.startswith("Target:"):
                current_target = stripped_line[len("Target:"):].strip().lower()
            elif stripped_line.startswith("Guessed:") and current_target is not None:
                guessed_value = stripped_line[len("Guessed:"):].strip().lower()

                total_predictions += 1
                if current_target == guessed_value:
                    correct_predictions += 1

                # Reset current_target after a pair is processed
                current_target = None

    if total_predictions == 0:
        return 0.0  # Avoid division by zero if no pairs were found
    else:
        #print(f"Total Predictions: {total_predictions}, Correct Predictions: {correct_predictions}")
        return correct_predictions / total_predictions


def score_dir(dirpath: str):
    """
    Takes a directory path and reads all .txt files in the directory,
    calls calculate_accuracy on each one, and stores the results in a dictionary.

    Args:
        dirpath (str): The path to the directory containing the text files.
    """
    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"The directory '{dirpath}' does not exist.")
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f"The path '{dirpath}' is not a directory.")

    accs = {}
    print(f"\nProcessing files in directory: {dirpath}")

    # Find all files in the dir
    for entry_name in os.listdir(dirpath):
        if entry_name.endswith(".txt"):
            #print(f"  - Processing file: {entry_name}")
            full_path = os.path.join(dirpath, entry_name)
            file_name_without_ext = os.path.splitext(entry_name)[0]
            accs[file_name_without_ext] = calculate_accuracy(full_path)

    print("\n--- Accuracy Scores Summary ---")
    for key in sorted(accs.keys()):
        print(f"{key}\t{accs[key]:.3f}")
    print("-----------------------------\n")

    # Compute Fragility Scores
    accs['noise-all'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], [accs['noise1'],accs['noise2'],accs['noise3'],accs['noise4'],accs['noise5']])] )
    accs['spell-all'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], [accs['noise-spell1'],accs['noise-spell2'],accs['noise-spell3'],accs['noise-spell4'],accs['noise-spell5']])] )
    accs['lookalike-noise-all'] = sum( [w*a for w,a in zip([0.35,0.25,0.2,0.15,0.05], [accs['noise-lookalike-noise1'],accs['noise-lookalike-noise2'],accs['noise-lookalike-noise3'],accs['noise-lookalike-noise4'],accs['noise-lookalike-noise5']])] )

    # Grade for all phone/digit tests.
    phone_grade = 0.2*accs['whatisthis-dig'] + \
        0.1*accs['plainphone'] + 0.1*accs['plainphone-CoD'] + 0.1*accs['plainphone-CoT'] + 0.1*accs['plainphone-CoT-CoD'] + \
        0.1*accs['plainphone-multi-CoT-CoD'] + \
        0.1*accs['noise-all'] + 0.1*accs['spell-all'] + 0.1*accs['lookalike-noise-all']

    # Grade for phone/digit but no CoD.
    phone_noCoD_grade = 0.2*accs['whatisthis-dig'] + \
        0.1*accs['plainphone'] + 0.2*accs['plainphone-CoT'] + \
        0.2*accs['plainphone-multi-CoT'] + \
        0.1*accs['noise-all'] + 0.1*accs['spell-all'] + 0.1*accs['lookalike-noise-all']

    # How much do unicode chars help -- compared to just underscore blanks.
    improvement = min(1, max(0, (accs['whatisthis-word'] - accs['whatisthis-word-under'])) / accs['whatisthis-word-under'])
    # Include the raw unicode-word score, but also its improvement over underscores.
    letters_grade = 0.2*accs['whatisthis-let'] + 0.3*accs['whatisthis-word'] + 0.2*improvement + 0.3*accs['whatisthis-word-noise2']

    # Need to put MMLU and one other NLP test here.
    mmlu_grade = min(1.0, accs['mmlu-20']/accs['mmlu'])
    NLU_grade = 0.6*mmlu_grade + 0.4*accs['whatisthis-sent']
    overall_grade = 0.3*NLU_grade + 0.3*letters_grade + 0.4*phone_grade

    model_name = dirpath.strip("/").split('/')[-1]
    print(f"PHONE grade for {model_name}         \t{round(phone_grade,3)}")
    print(f"PHONE grade (no CoD) for {model_name}\t{round(phone_noCoD_grade,3)}")
    print(f"LETTERS grade for {model_name}       \t{round(letters_grade,3)}")
    print(f"MMLU grade for {model_name}          \t{round(mmlu_grade,3)}")
    print(f"NLU grade for {model_name}           \t{round(NLU_grade,3)}")
    print(f"OVERALL grade for {model_name}       \t{round(overall_grade,3)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate accuracy from text files.")
    parser.add_argument("dirpath", type=str, help="Path to the directory containing text files.")
    args = parser.parse_args()

    score_dir(args.dirpath)
