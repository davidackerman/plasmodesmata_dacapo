# %%
import subprocess
import yaml


def find_line_numbers_via_grep(file_path, keywords):
    line_numbers = {}
    for keyword in keywords:
        # Use `grep -n` to find lines with the keyword and exclude `null` values
        result = subprocess.run(
            ["grep", "-n", f"{keyword}:", file_path], capture_output=True, text=True
        )
        # Filter out lines where the keyword is followed by `null`
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            filtered_lines = [
                int(line.split(":")[0])
                for line in lines
                if f"{keyword}: null" not in line  # Exclude lines with `null`
            ]
            line_numbers[keyword] = filtered_lines
        else:
            line_numbers[keyword] = []  # No matches found
    return line_numbers


def read_except_section(file_path, start_line, end_line):
    result_lines = []
    with open(file_path, "r") as file:
        for idx, line in enumerate(file, start=1):
            if start_line <= idx <= end_line:
                continue  # Skip lines between start_line and end_line
            result_lines.append(line)

    return "".join(result_lines)


file_path = "/nrs/cellmap/ackermand/cellmap_experiments/configs/runs/finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.0003_bs_6__0.yaml"
keywords = ["sample_points", "validate_configs"]  # Updated to validate_configs

# Find the line numbers
line_numbers = find_line_numbers_via_grep(file_path, keywords)

# Get the first `sample_points` and the first `validate_configs` line
sample_points_line = line_numbers.get("sample_points", [None])[0]
validate_configs_line = line_numbers.get("validate_configs", [None])[0]

if sample_points_line is not None and validate_configs_line is not None:
    # Read the YAML excluding the lines between sample_points and validate_configs
    # start from one before validate_configs_line since want to include weights
    result_yaml = yaml.full_load(
        read_except_section(file_path, sample_points_line, validate_configs_line - 1)
    )
    # pretty print yaml file

    pretty_yaml = yaml.dump(
        result_yaml, sort_keys=False, default_flow_style=False
    )  # Pretty print YAML

    print(pretty_yaml)
else:
    print("Could not find the required sections.")


import dacao
# %%
