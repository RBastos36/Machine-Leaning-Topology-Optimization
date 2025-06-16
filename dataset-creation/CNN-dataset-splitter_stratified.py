# Author: Ricardo A. O. Bastos
# Created: June 2025

import h5py
import json
import numpy as np
from collections import defaultdict
import re


def print_hdf5_structure(hdf5_file, group_path="/", level=0):
    """Print the structure of HDF5 file for debugging."""
    group = hdf5_file[group_path]
    indent = "  " * level

    print(f"\n{indent}Contents of: {group_path}")
    for name, item in group.items():
        full_path = f"{group_path}/{name}" if group_path != "/" else f"/{name}"
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {full_path}")
            print_hdf5_structure(hdf5_file, full_path, level + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}Dataset: {full_path}, Shape: {item.shape}")


def check_paths(hdf5_file, problem_path, iter_path):
    """Check which paths exist and which don't."""
    required_paths = {
        'fixed_x': f"{problem_path}/setup/constraints/fixed_x",
        'fixed_y': f"{problem_path}/setup/constraints/fixed_y",
        'loads_x': f"{problem_path}/setup/loads/x",
        'loads_y': f"{problem_path}/setup/loads/y",
        'domain': f"{problem_path}/{iter_path}/domain",
        'displacement_x': f"{problem_path}/{iter_path}/displacements/x",
        'displacement_y': f"{problem_path}/{iter_path}/displacements/y"
    }

    print(f"\nChecking paths for: {iter_path}")
    exists = {}
    for name, path in required_paths.items():
        exists[name] = path in hdf5_file
        print(f"  {name}: {'✓' if exists[name] else '✗'} ({path})")

    return exists, required_paths


def extract_metadata_from_problem_name(problem_name):
    """Extract metadata parameters from the problem name using regex.
    In this dataset:
    - "dir" is the horizontal magnitude
    - "mag" is the vertical magnitude
    """
    metadata = {}

    # Example pattern: vf0.50_pos0.50_dir-6.123233995736766e-17_mag1.0_nelx180_nely60_rmin5.4
    vf_match = re.search(r'vf(\d+\.\d+)', problem_name)
    if vf_match:
        metadata["volfrac"] = float(vf_match.group(1))

    pos_match = re.search(r'pos(\d+\.\d+)', problem_name)
    if pos_match:
        metadata["load_position"] = float(pos_match.group(1))

    # Extract horizontal load magnitude (dir parameter)
    dir_match = re.search(r'dir([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', problem_name)
    if dir_match:
        metadata["load_magnitude_horizontal"] = float(dir_match.group(1))

    # Extract vertical load magnitude (mag parameter)
    mag_match = re.search(r'mag(\d+\.\d+)', problem_name)
    if mag_match:
        metadata["load_magnitude_vertical"] = float(mag_match.group(1))

    nelx_match = re.search(r'nelx(\d+)', problem_name)
    if nelx_match:
        metadata["nelx"] = int(nelx_match.group(1))

    nely_match = re.search(r'nely(\d+)', problem_name)
    if nely_match:
        metadata["nely"] = int(nely_match.group(1))

    rmin_match = re.search(r'rmin(\d+\.\d+)', problem_name)
    if rmin_match:
        metadata["rmin"] = float(rmin_match.group(1))

    return metadata


def get_all_samples(hdf5_file, problem_name, debug=True):
    """Extract complete HDF5 paths for all iterations of a given problem."""
    try:
        problem_path = f"problems/{problem_name}"
        if debug:
            print(f"\nExamining problem: {problem_name}")
            print("Available structure:")
            print_hdf5_structure(hdf5_file, problem_path)

        problem_group = hdf5_file[problem_path]

        # Extract metadata from problem name
        metadata = extract_metadata_from_problem_name(problem_name)
        if debug and metadata:
            print(f"Extracted metadata: {metadata}")

        # Get all iteration numbers
        iter_groups = [g for g in problem_group.keys() if g.startswith("iter_")]
        if not iter_groups:
            if debug:
                print(f"No iteration groups found for problem {problem_name}")
                print(f"Available groups: {list(problem_group.keys())}")
            raise ValueError(f"No iteration groups found for problem {problem_name}")

        if debug:
            print(f"\nFound iterations: {iter_groups}")

        samples = []
        for iter_group in iter_groups:
            iter_num = int(iter_group.split("_")[1])

            path_exists = {}
            paths = {}

            if debug:
                print(f"\nProcessing iteration: {iter_group}")
                path_exists, paths = check_paths(hdf5_file, problem_path, iter_group)

            # Skip iteration if any required path is missing
            if not all(path_exists.values()):
                if debug:
                    print(f"Skipping iteration {iter_num} - missing required paths")
                continue

            # Combine problem metadata with iteration info
            sample_metadata = {
                "problem_name": problem_name,
                "iteration": iter_num
            }
            if metadata:
                sample_metadata.update(metadata)

            sample = {
                "inputs": {
                    "fixed_x": paths['fixed_x'],
                    "fixed_y": paths['fixed_y'],
                    "loads_x": paths['loads_x'],
                    "loads_y": paths['loads_y'],
                    "domain": paths['domain']
                },
                "outputs": {
                    "displacement_x": paths['displacement_x'],
                    "displacement_y": paths['displacement_y']
                },
                "metadata": sample_metadata
            }

            samples.append(sample)

        return samples

    except Exception as e:
        if debug:
            print(f"Error processing problem {problem_name}: {str(e)}")
        raise


def split_dataset(hdf5_path, output_json_path, train_size=0.8, validation_size=0.15):
    """Stratified dataset splitting for better representation in train, validation, and test sets."""

    with h5py.File(hdf5_path, 'r') as f:
        print("\nAnalyzing HDF5 file structure...")

        problem_names = list(f["problems"].keys())
        print(f"\nFound {len(problem_names)} problems: {problem_names}")

        # Dictionary to group samples by parameter combinations
        grouped_samples = defaultdict(list)
        all_samples = []

        for name in problem_names:
            try:
                problem_samples = get_all_samples(f, name, debug=True)
                all_samples.extend(problem_samples)
                print(f"Added {len(problem_samples)} samples from problem {name}")

                for sample in problem_samples:
                    metadata = sample["metadata"]

                    # Extract key parameters for stratification
                    key_parts = []
                    for param in ["volfrac", "load_position", "load_magnitude_vertical", "load_magnitude_horizontal"]:
                        if param in metadata:
                            key_parts.append(metadata[param])
                        else:
                            # Use None as placeholder if parameter is missing
                            key_parts.append(None)

                    # Create a tuple key for grouping
                    key = tuple(key_parts)
                    grouped_samples[key].append(sample)

            except Exception as e:
                print(f"Error processing problem {name}: {str(e)}")
                continue

        total_samples = len(all_samples)
        print(f"\nTotal valid samples (including all iterations): {total_samples}")
        print(f"Number of different parameter combinations: {len(grouped_samples)}")

        if total_samples == 0:
            raise ValueError("No valid samples found in the dataset")

        train_samples, validation_samples, test_samples = [], [], []

        # Stratified split - ensure proportional representation for each parameter combination
        np.random.seed(24)
        for param_combination, samples in grouped_samples.items():
            print(f"\nParameter combination: {param_combination}, Samples: {len(samples)}")
            np.random.shuffle(samples)  # Shuffle within the group

            n_total = len(samples)
            n_train = max(int(n_total * train_size), 1)
            n_validation = max(int(n_total * validation_size), 1)

            n_validation = min(n_validation, n_total - n_train)
            n_test = n_total - n_train - n_validation

            train_samples.extend(samples[:n_train])
            validation_samples.extend(samples[n_train:n_train + n_validation])
            test_samples.extend(samples[n_train + n_validation:])

            # Print the distribution for this parameter combination
            print(f"  - Training: {n_train}/{n_total} ({n_train / n_total:.1%})")
            print(f"  - Validation: {n_validation}/{n_total} ({n_validation / n_total:.1%})")
            print(f"  - Test: {n_test}/{n_total} ({n_test / n_total:.1%})")

        dataset_info = {
            'train': train_samples,
            'validation': validation_samples,
            'test': test_samples,
            'metadata': {
                'hdf5_file': hdf5_path,
                'total_samples': len(train_samples) + len(validation_samples) + len(test_samples),
                'train_size': len(train_samples),
                'validation_size': len(validation_samples),
                'test_size': len(test_samples),
                'parameter_combinations': len(grouped_samples)
            }
        }

        print("\nDataset Summary:")
        print(f"Total samples: {total_samples}")
        print(f"- {len(train_samples)} training samples ({len(train_samples) / total_samples:.1%})")
        print(f"- {len(validation_samples)} validation samples ({len(validation_samples) / total_samples:.1%})")
        print(f"- {len(test_samples)} test samples ({len(test_samples) / total_samples:.1%})")

        with open(output_json_path, 'w') as json_file:
            json.dump(dataset_info, json_file, indent=2, sort_keys=True)

        return dataset_info


if __name__ == "__main__":
    hdf5_path = "TESTING_NO_PENAL.h5"
    output_json_path = "SPLIT_TESTING_NO_PENAL.json"

    try:
        dataset_info = split_dataset(hdf5_path, output_json_path)
        print("\nDataset split completed successfully!")
    except Exception as e:
        print(f"\nError splitting dataset: {str(e)}")