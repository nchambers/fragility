# 
# This is a one-time program that reads a directory of output text files from fragility.py
# It creates JSON datasets from these files.
# If people don't want to run fragility.py, they can use these JSON files instead.
#
# It assumes each TXT file in the directory contains lines formatted as:
#   Target: the goal output
#   Given: the input that was used to generate the guess
#
# This program reads each file, extracts the target and given/input values,
# and then outputs these into a JSON file as a list of dictionaries.
#

import os
import json
import argparse

def extract_dataset(dirpath: str, output_dir: str):
    """
    Reads all .txt files in the specified directory, extracts the target 
    and given values, and writes them to a JSON file.

    Args:
        dirpath (str): The path to the directory containing the text files.
        output_dir (str): The path to the directory to save the output JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"The directory '{dirpath}' does not exist.")


    for filename in os.listdir(dirpath):
        if filename.endswith('.txt'):
            dataset = []
            file_path = os.path.join(dirpath, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                current_target = None
                current_given = None

                for line in f:
                    stripped_line = line.strip()
                    if stripped_line.startswith("Target:"):
                        current_target = stripped_line[len("Target:"):].strip()
                    elif stripped_line.startswith("Given:"):
                        current_given = stripped_line[len("Given:"):].strip()

                    if current_target is not None and current_given is not None:
                        dataset.append({
                            "target": current_target,
                            "challenge": current_given
                        })
                        # Reset for next entry
                        current_target = None
                        current_given = None

            output_file = os.path.join(output_dir, f'{filename[0:-4]}.json')
            with open(output_file, 'w', encoding='utf-8') as out_file:
                json.dump(dataset, out_file, indent=4)
            print(f"Dataset extracted to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract dataset from text files.")
    parser.add_argument('dirpath', type=str, help='Path to the directory containing text files.')
    parser.add_argument('output_dir', type=str, help='Path to the output dir for the JSON files.')

    args = parser.parse_args()
    extract_dataset(args.dirpath, args.output_dir)
