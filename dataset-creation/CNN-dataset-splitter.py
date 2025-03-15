#!/usr/bin/env python3

import h5py
import json
import numpy as np


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


def get_all_samples(hdf5_file, problem_name, debug=True):
    """Extract complete HDF5 paths for all iterations of a given problem."""
    try:
        problem_path = f"problems/{problem_name}"
        if debug:
            print(f"\nExamining problem: {problem_name}")
            print("Available structure:")
            print_hdf5_structure(hdf5_file, problem_path)

        problem_group = hdf5_file[problem_path]

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

            if debug:
                print(f"\nProcessing iteration: {iter_group}")
                path_exists, paths = check_paths(hdf5_file, problem_path, iter_group)

            # Skip iteration if any required path is missing
            if not all(path_exists.values()):
                if debug:
                    print(f"Skipping iteration {iter_num} - missing required paths")
                continue

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
                "metadata": {
                    "problem_name": problem_name,
                    "iteration": iter_num
                }
            }

            samples.append(sample)

        return samples

    except Exception as e:
        if debug:
            print(f"Error processing problem {problem_name}: {str(e)}")
        raise


def split_dataset(hdf5_path, output_json_path, train_size=0.7, validation_size=0.15):
    """Split the dataset into train, validation, and test sets."""
    with h5py.File(hdf5_path, 'r') as f:
        print("\nAnalyzing HDF5 file structure...")
        print_hdf5_structure(f, "problems")

        # Get all problem names
        problem_names = list(f["problems"].keys())
        print(f"\nFound {len(problem_names)} problems: {problem_names}")

        # Get all samples from all problems
        all_samples = []
        for name in problem_names:
            try:
                problem_samples = get_all_samples(f, name, debug=True)
                all_samples.extend(problem_samples)
                print(f"Added {len(problem_samples)} samples from problem {name}")
            except Exception as e:
                print(f"Error processing problem {name}: {str(e)}")
                continue

        total_samples = len(all_samples)
        print(f"\nTotal valid samples (including all iterations): {total_samples}")

        if total_samples == 0:
            raise ValueError("No valid samples found in the dataset")

        # Calculate split sizes
        n_train = max(int(total_samples * train_size), 1)
        n_validation = max(int(total_samples * validation_size), 1)
        n_test = total_samples - n_train - n_validation

        # Shuffle samples
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_samples)

        # Split samples
        train_samples = all_samples[:n_train]
        validation_samples = all_samples[n_train:n_train + n_validation]
        test_samples = all_samples[n_train + n_validation:]

        # Create dataset info
        dataset_info = {
            'train': train_samples,
            'validation': validation_samples,
            'test': test_samples,
            'metadata': {
                'hdf5_file': hdf5_path,
                'total_samples': total_samples,
                'train_size': len(train_samples),
                'validation_size': len(validation_samples),
                'test_size': len(test_samples)
            }
        }

        # Print summary
        print(f"\nDataset Summary:")
        print(f"Total samples: {total_samples}")
        print(f"- {len(train_samples)} training samples ({len(train_samples) / total_samples:.1%})")
        print(f"- {len(validation_samples)} validation samples ({len(validation_samples) / total_samples:.1%})")
        print(f"- {len(test_samples)} test samples ({len(test_samples) / total_samples:.1%})")

        # Save to JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump(dataset_info, indent=2, sort_keys=True, fp=json_file)

        return dataset_info


if __name__ == "__main__":
    # Parameters
    hdf5_path = "cantilever-diagonal_dataset.h5"  # Update this to your file path
    output_json_path = "dataset_split.json"

    # Split dataset
    try:
        dataset_info = split_dataset(hdf5_path, output_json_path, train_size=0.7, validation_size=0.15)
        print("\nDataset split completed successfully!")
    except Exception as e:
        print(f"\nError splitting dataset: {str(e)}")
