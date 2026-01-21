# Create the binary head dataset for the evo project

from itertools import zip_longest
import datasets
import numpy as np
import torch
from transformers import AutoTokenizer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Dataset configurations
DATASETS_CONFIG = [
    {
        'input_file': f'{ROOT_DIR}/crispr-datasets/output_crispr_data.csv',
        'output_dir': f'{ROOT_DIR}/crispr-datasets/evo/multi_data_only_bona_fide',
        'name': 'Main CRISPR Dataset'
    },
    {
        'input_file': f'{ROOT_DIR}/data/simulated_reads_meta.csv',
        'output_dir': f'{ROOT_DIR}/data/simulated_reads_meta_processed',
        'name': 'Simulated Reads Dataset'
    }
]

model_name = 'togethercomputer/evo-1-8k-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "X"

def calc_seq_and_labels(row):
    """
    Calculate the sequence and repeat spacer sequence
    :param row: row of the dataset object
    :return: a tuple containing the sequence and repeat spacer sequence (label)
    """
    labels = row["labels"]
    seq = row["sequence"]

    # Determine the amount of leading 2's in the labels
    leading_twos = len(labels) - len(labels.lstrip('2'))

    # Determine the amount of trailing 2's in the labels
    trailing_twos = len(labels) - len(labels.rstrip('2'))

    # Take a random number between 0 and the minimum of the leading_twos and 500
    if leading_twos == 0:
        extra_leading_nt = 0
    else:
        extra_leading_nt = torch.randint(0, min(leading_twos, 501), (1,)).item()

    # Take a random number between 0 and the minimum of the trailing_twos and 500
    if trailing_twos == 0:
        extra_trailing_nt = 0
    else: 
        extra_trailing_nt = torch.randint(0, min(trailing_twos, 501), (1,)).item()

    # Calculate the start position using the extra_leading_nt
    start = leading_twos - extra_leading_nt

    # Calculate the end position using the extra_trailing_nt
    end = len(labels) - (trailing_twos - extra_trailing_nt)

    # Extract the sequence using the start and end positions
    seq = seq[start:end]

    # Extract the labels using the start and end positions
    labels = labels[start:end]

    # Convert the labels to a list of integers
    labels = [int(label) for label in labels]
        
    return {'sequence': seq, 'labels': labels}

def tokenize_data(sample, max_length=8000):
    tokenized = tokenizer(sample['sequence'], padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    labels = sample['labels']
    # Pad or truncate orf_labels to match the tokenized sequence length
    if len(labels) < max_length:
        labels += [-100] * (max_length - len(labels))
    else:
        labels = labels[:max_length]

    tokenized['labels'] = torch.tensor(labels, dtype=torch.long)

    # Remove batch dimension added by tokenizer
    for key in tokenized:
        tokenized[key] = tokenized[key].squeeze(0)

    return tokenized

def process_dataset(input_file, output_dir, dataset_name):
    """Process a single dataset from CSV to train/val/test splits."""
    print("=" * 80)
    print(f"Processing: {dataset_name}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Load dataset
    dataset = datasets.load_dataset('csv', data_files=input_file)
    
    # Filter out the sequences that are not bona fide
    dataset = dataset.filter(lambda x: x['category'] == 'Bona-fide')
    
    # Rename mask column to labels
    dataset = dataset.rename_column('mask', 'labels')
    
    # Rename acc_num to acc_number
    dataset = dataset.rename_column('acc_num', 'acc_number')
    
    # Apply the sequence and label calculation
    dataset = dataset.map(calc_seq_and_labels)
    
    # Keep only the sequence and label columns
    dataset = dataset.select_columns(['acc_number', 'start', 'end', 'sequence', 'labels'])
    
    # Tokenize the dataset
    dataset = dataset.map(tokenize_data)
    
    # Split into train/validation/test
    train_validtest = dataset["train"].train_test_split(test_size=0.30, seed=42)
    valid_test = train_validtest["test"].train_test_split(test_size=2/3, seed=42)
    
    # Assign the final datasets
    dataset = datasets.DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"]
    })
    
    # Print initial statistics
    total_sequences = len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
    train_percentage = len(dataset['train']) / total_sequences * 100
    validation_percentage = len(dataset['validation']) / total_sequences * 100
    test_percentage = len(dataset['test']) / total_sequences * 100
    
    print(f"\nInitial split:")
    print(f"  Train: {len(dataset['train'])} ({train_percentage:.2f}%)")
    print(f"  Validation: {len(dataset['validation'])} ({validation_percentage:.2f}%)")
    print(f"  Test: {len(dataset['test'])} ({test_percentage:.2f}%)")
    
    # Remove overlapping sequences
    train_seq_set = set(tuple(seq) for seq in dataset['train']['input_ids'])
    validation_seq_set = set(tuple(seq) for seq in dataset['validation']['input_ids'])
    test_seq_set = set(tuple(seq) for seq in dataset['test']['input_ids'])
    
    # Find overlaps
    train_validation_overlap = train_seq_set.intersection(validation_seq_set)
    train_test_overlap = train_seq_set.intersection(test_seq_set)
    validation_test_overlap = validation_seq_set.intersection(test_seq_set)
    
    if len(train_validation_overlap) > 0 or len(train_test_overlap) > 0 or len(validation_test_overlap) > 0:
        print(f"\nRemoving overlapping sequences:")
        print(f"  Train-Validation overlap: {len(train_validation_overlap)}")
        print(f"  Train-Test overlap: {len(train_test_overlap)}")
        print(f"  Validation-Test overlap: {len(validation_test_overlap)}")
        
        # Remove overlaps
        dataset['train'] = dataset['train'].filter(lambda x: tuple(x['input_ids']) not in train_validation_overlap)
        dataset['train'] = dataset['train'].filter(lambda x: tuple(x['input_ids']) not in train_test_overlap)
        dataset['validation'] = dataset['validation'].filter(lambda x: tuple(x['input_ids']) not in validation_test_overlap)
        
        total_removed = len(train_validation_overlap) + len(train_test_overlap) + len(validation_test_overlap)
        print(f"  Total sequences removed: {total_removed}")
    
    # Print final statistics
    total_sequences = len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
    train_percentage = len(dataset['train']) / total_sequences * 100
    validation_percentage = len(dataset['validation']) / total_sequences * 100
    test_percentage = len(dataset['test']) / total_sequences * 100
    
    print(f"\nFinal split:")
    print(f"  Train: {len(dataset['train'])} ({train_percentage:.2f}%)")
    print(f"  Validation: {len(dataset['validation'])} ({validation_percentage:.2f}%)")
    print(f"  Test: {len(dataset['test'])} ({test_percentage:.2f}%)")
    
    # Save the dataset
    dataset.save_to_disk(output_dir)
    print(f"\n✓ Dataset saved to: {output_dir}")
    print("=" * 80 + "\n")
    
    return dataset

# Process all datasets
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CRISPR Dataset Processing Pipeline")
    print("=" * 80 + "\n")
    
    for config in DATASETS_CONFIG:
        try:
            process_dataset(
                input_file=config['input_file'],
                output_dir=config['output_dir'],
                dataset_name=config['name']
            )
        except FileNotFoundError as e:
            print(f"⚠ Warning: Could not find {config['input_file']}")
            print(f"  Skipping {config['name']}")
            print("=" * 80 + "\n")
            continue
        except Exception as e:
            print(f"✗ Error processing {config['name']}: {e}")
            print("=" * 80 + "\n")
            continue
    
    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)