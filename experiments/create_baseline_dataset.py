import random

folder = "data"

# File paths
source_path = "val_hqa_hq.txt"
target_path = "val_c.txt"
baseline_path = "val_hqa_hq_c.txt"

# count the number of target lines (images)
with open(f'{folder}\{target_path}', 'r') as target:
    # Read the all the lines in the target file
    target_lines = target.readlines()

    # Open the source file to count the number of lines
    with open(f'{folder}\{source_path}', 'r') as source:
        # Read the all the lines in the source file
        source_lines = source.readlines()

        # Shuffle the source lines
        random.shuffle(source_lines)

        # Select the first num_lines_target lines
        lines = source_lines[:len(source_lines) - len(target_lines)]

        # Append the target lines to the source lines
        lines.extend(target_lines)

        # write the remaining lines to a new file and add the target lines
        with open(f'{folder}\{baseline_path}', 'w') as baseline:
            baseline.writelines(lines)

# Print the number of lines for each file
print(f"Number of lines in {target_path}: {len(target_lines)}")
print(f"Number of lines in {source_path}: {len(source_lines)}")
print(f"Number of lines in {baseline_path}: {len(lines)}")

assert len(lines) == len(source_lines), "The number of lines in the baseline file should be equal to the number of lines in the source file"

