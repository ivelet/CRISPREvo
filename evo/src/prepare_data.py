# Create the dataset for the evo project

from itertools import zip_longest
import datasets
import torch
from transformers import AutoTokenizer

# Load in the updated_crispr_cas csv datafile into a datasets object
dataset = datasets.load_dataset('csv', data_files='/work/dlclarge2/koeksalr-crispr/crispr-datasets/updated_crispr_cas.csv')

# Print out number of rows in the dataset
print(f"Number of rows in the dataset (Bona-fide, Possible): {len(dataset['train'])}")

# Define the maximun length
max_length = 8000

# Create two new columns in the dataset object containing the input sequence and the corresponding output
def create_seq_and_output(row, max_length=max_length):
    seq = ''
    output = ''

    # Check if the sequence has a cassette
    start = row["cas_cas_start"]
    end = row["cas_cas_end"]
    if start is not None and end is not None:
        start = int(start)
        end = int(end)
        
        # Open the fasta file containing the cassette sequence
        fasta_file = f'/work/dlclarge2/koeksalr-crispr/crispr-datasets/bona_fide_sequences/{row["acc_number"]}.fasta'
        try: 
            with open(fasta_file, 'r') as f:
                record = f.read()

            # Extract the cassette sequence
            # Find the first newline character and take sequence from the next character
            record = record[record.find('\n')+1:]
            # Delete all newline characters
            record = record.replace('\n', '')
            # Extract the cassette sequence
            seq += record[start:end]
            output += '!' * len(seq)

        except Exception as e:
            print(f"Error while processing file: {fasta_file}")
            raise e

    # Divide the strings into lists
    repeats = row["repeat_sequences"].split(' ')
    spacers = row["spacer_sequences"].split(' ')

    # Create the sequence and output
    for i in range(len(repeats)):
        # Add the start position of the repeat to the output
        output += '-' * len(repeats[i])

        # Add the repeat sequence to the sequence
        seq += repeats[i]

        # If we have reached the last repeat there is no spacer following it
        if i >= len(spacers):
            break

        # Add the start position of the spacer to the labels
        output += '+' * len(spacers[i])

        # Add the spacer sequence to the sequence
        seq += spacers[i]

    # If the sequence is longer than the maximum length, truncate it
    if len(seq) > max_length:
        seq = seq[:max_length]
        output = output[:max_length]
        
    # Add the sequence and the output to the row (stored as labels, since the trainer expects that name)
    return {'sequence': seq + '|' + output}

# Apply the function to the dataset object
dataset = dataset.map(create_seq_and_output)


# Keep the sequence column
dataset = dataset.select_columns('sequence')

# Tokenize the sequences using the Evo model
model_name = 'togethercomputer/evo-1-8k-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = 'X'
tokenizer.sep_token = '|'

def tokenize_data(sample, max_length=max_length):
    tokenized = tokenizer(sample['sequence'], padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')

    # Remove batch dimension added by tokenizer
    for key in tokenized:
        tokenized[key] = tokenized[key].squeeze(0)

    # Add labels which are the tokenized input_ids
    tokenized['labels'] = tokenized['input_ids'].clone()

    return tokenized

# Tokenize the dataset
dataset = dataset.map(tokenize_data, batched=False, num_proc=12)

# Create a 75-10-15 train-validation-test split, then remove overlapping sequences between the train, validation, and test sets trying to land on a 70-10-20 split
# Perform the first split (75-15 train-test split)
train_validtest = dataset["train"].train_test_split(test_size=0.25, seed=42)
# Split the validation+test set into a validation and test set (60-15 validation-test split)
valid_test = train_validtest["test"].train_test_split(test_size=0.55, seed=42)
# Assign the final datasets
dataset = datasets.DatasetDict({
    "train": train_validtest["train"],
    "validation": valid_test["train"],
    "test": valid_test["test"]
})

# Print the number of sequences in the train, validation, and test sets
# Calculate the percentage of sequences in the train, validation, and test sets
total_sequences = len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
train_percentage = len(dataset['train']) / total_sequences * 100
validation_percentage = len(dataset['validation']) / total_sequences * 100
test_percentage = len(dataset['test']) / total_sequences * 100

print(f"Number of sequences in the train set: {len(dataset['train'])} ({train_percentage:.2f}%)")
print(f"Number of sequences in the validation set: {len(dataset['validation'])} ({validation_percentage:.2f}%)")
print(f"Number of sequences in the test set: {len(dataset['test'])} ({test_percentage:.2f}%)\n")

# If the train, validation, and test sets aren't disjoint, remove the overlapping sequence with priority given to the test set, then the validation set

# Create set of input ids for each dataset (input ids in case truncated sequences become identical)
train_seq_set = set(tuple(seq) for seq in dataset['train']['input_ids'])
validation_seq_set = set(tuple(seq) for seq in dataset['validation']['input_ids'])
test_seq_set = set(tuple(seq) for seq in dataset['test']['input_ids'])

# If there are overlapping sequences print out that we are removing them with priority given to the test set, then the validation set
if len(train_seq_set.intersection(validation_seq_set)) > 0 or len(train_seq_set.intersection(test_seq_set)) > 0 or len(validation_seq_set.intersection(test_seq_set)) > 0:
    print("Removing overlapping sequences with priority given to the test set, then the validation set.\n")

# Find the overlapping sequences between the train and validation sets
train_validation_overlap = train_seq_set.intersection(validation_seq_set)
# Find the overlapping sequences between the train and test sets
train_test_overlap = train_seq_set.intersection(test_seq_set)
# Find the overlapping sequences between the validation and test sets
validation_test_overlap = validation_seq_set.intersection(test_seq_set)

# Print the number of overlapping sequences between the train and validation sets
print(f"Number of overlapping sequences between the train and validation sets: {len(train_validation_overlap)}")
# Print the number of overlapping sequences between the train and test sets
print(f"Number of overlapping sequences between the train and test sets: {len(train_test_overlap)}")
# Print the number of overlapping sequences between the validation and test sets
print(f"Number of overlapping sequences between the validation and test sets: {len(validation_test_overlap)}\n")

# Remove the overlapping sequences between the train and validation sets
dataset['train'] = dataset['train'].filter(lambda x: tuple(x['input_ids']) not in train_validation_overlap)
# Remove the overlapping sequences between the train and test sets
dataset['train'] = dataset['train'].filter(lambda x: tuple(x['input_ids']) not in train_test_overlap)
# Remove the overlapping sequences between the validation and test sets
dataset['validation'] = dataset['validation'].filter(lambda x: tuple(x['input_ids']) not in validation_test_overlap)

# Print total number of sequences in the train, validation, and test sets after removing the overlapping sequences
print(f"Total number of sequences in the train, validation, and test sets after removing overlapping sequences: {len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])}")
# Print total number of sequences removed
print(f"Total number of sequences removed: {len(train_validation_overlap) + len(train_test_overlap) + len(validation_test_overlap)}\n")

# Print the number of sequences in the train, validation, and test sets after removing the overlapping sequences

# Calculate the percentage of sequences in the train, validation, and test sets after removing the overlapping sequences
total_sequences = len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])
train_percentage = len(dataset['train']) / total_sequences * 100
validation_percentage = len(dataset['validation']) / total_sequences * 100
test_percentage = len(dataset['test']) / total_sequences * 100

print(f"Number of sequences in the train set after removing overlapping sequences: {len(dataset['train'])} ({train_percentage:.2f}%)")
print(f"Number of sequences in the validation set after removing overlapping sequences: {len(dataset['validation'])} ({validation_percentage:.2f}%)")
print(f"Number of sequences in the test set after removing overlapping sequences: {len(dataset['test'])} ({test_percentage:.2f}%)\n")

# Save the dataset as a datasets object
dataset.save_to_disk('/work/dlclarge2/koeksalr-crispr/crispr-datasets')
