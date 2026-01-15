# Create the binary head dataset for the evo project

from itertools import zip_longest
import datasets
import torch
from transformers import AutoTokenizer

# Load in the updated_crispr_cas csv datafile into a datasets object
dataset = datasets.load_dataset('csv', data_files='/home/fr/fr_fr/fr_ls1369/master-thesis/updated_crispr_cas.csv')

# Filter out the sequences that are not bona fide
dataset = dataset.filter(lambda x: x['category'] == 'Bona-fide')

# Create a sequence column by concatenating the repeat and spacer sequences
def create_seq(row):
    """
    Calculate the sequence and repeat spacer sequence
    :param row: row of the dataset object
    :return: a tuple containing the sequence and repeat spacer sequence (label)
    """
    # Divide the strings into lists
    repeats = row["repeat_sequences"].split(' ')
    spacers = row["spacer_sequences"].split(' ')
    seq = ''
    for repeat, spacer in zip_longest(repeats, spacers, fillvalue=""):
        seq += repeat + spacer
        
    return {'sequence': seq}

# Apply the function to the dataset object
dataset = dataset.map(create_seq)

# Encode the cas_Cassete type as a label
dataset = dataset.class_encode_column("cas_Cassete type", "label")

print(dataset.features)

# Keep only the sequence and label columns
dataset = dataset.select_columns(['acc_number', 'start', 'end','sequence', 'label'])

model_name = 'togethercomputer/evo-1-8k-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "X"

def tokenize_data(sample, max_length=8000):
    tokenized = tokenizer(sample['sequence'], padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
    labels = sample['labels']
    # Pad or truncate orf_labels to match the tokenized sequence length
    if len(labels) < max_length:
        labels += [-100] * (max_length - len(labels))
    else:
        labels = labels[:max_length]

    tokenized['labels'] = torch.tensor(labels, dtype=torch.float32)

    # Remove batch dimension added by tokenizer
    for key in tokenized:
        tokenized[key] = tokenized[key].squeeze(0)

    return tokenized

# Tokenize the dataset
dataset = dataset.map(tokenize_data, batched=False, num_proc=12)

# Create a 75-10-15 train-validation-test split, then remove overlapping sequences between the train, validation, and test sets trying to land on a 70-10-20 split
# Perform the first split (75-15 train-test split)
train_validtest = dataset["train"].train_test_split(test_size=0.25, seed=42)
# Split the validation+test set into a validation and test set (60-15 validation-test split)
valid_test = train_validtest["test"].train_test_split(test_size=0.60, seed=42)
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
dataset.save_to_disk('/work/dlclarge2/koeksalr-crispr/datasets/evo/bin_data_only_bona_fide')
