import sys
from model import EvoBinaryTokenClassificationModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from itertools import chain, zip_longest
from datasets import Dataset
import torch
from eval_metrics import compute_metrics

if __name__ == "__main__":
    # If argument is provided, set the model id
    # Otherwise, use the default model
    model_id = None
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    # If the second argument is provided, set the number of samples
    # Otherwise, use the default amount of samples
    num_samples = 5
    if len(sys.argv) > 2: 
        num_samples = int(sys.argv[2])

    # Load the saved model
    model_dir = "/work/dlclarge2/koeksalr-crispr/results/bin/model"
    if model_id is not None:
        model_dir += f"-{model_id}"

    model = EvoBinaryTokenClassificationModel.from_pretrained(model_dir)

    # Create a dataframe with the data
    test_dataset = pd.DataFrame([{
        'list_repeats': ['GGCTCTCCCCCGCGCACGCGCGGGTACACCG', 'ATCTCTCCCCCGCGCACGCGCGGGTACACCC', 'CCCATTTTCTTCCCCGCGCGCGGGTGAACCG', 'ATCTGTCCTCCCCGCGCGCGCGGGTGAACCG', 'AGCTCTGTCTTCCCCGCGCGCGGGTGAACCC', 'CTGTCTCCTCCGCGCACGCGCGGGTGAACCG', 'CTGTCTCCTCCGCGCACGCGCGGGTGAACCG', 'TCTCCTATATTCCCCGCGCGCGGGTGAACCG', 'GACTGTCGTCCCCGTGTGCGCGGGTAACCCG', 'GGATCTGTCTTCCCCGCGCGCGGGTGAACCA', 'GCCTGTCTTCCCCGCGTGCGCGGGTGATCCA', 'ACTCTCGCACCGAGCACGCGCGGGTAAGCCC', 'AGCCCTCTCTCCCGCGCACGCGGGTAAACCC', 'AATCGCTCCCCGCTCACATGCGGGTAAACCC', 'ACTCTTCCTCCGCAAGCATGCGGGTGAGCCT', 'CTTGCTTCCTGGGCACATGCGGGTGAGCA', 'GCTTCTCTTCGGCATGTATGTGGGTTGCCAG', 'CGAGTCTTCCCGCATACGCGCGGGTAATCCG'],
        'list_spacers': ['TCGTGGACGTCTGTCGGCGTACCCTGCAGG', 'TGCGACGAGTTCGAGCGCGGCAGCGC', 'GTCGAGATCATACGCTCCGCCCATCCGTCC', 'ATGTTCACCAAGGATGTCAGCCTGACCC', 'ACTCCCTCGCGATGTTTGAGCACGGGTGTTAT', 'ACCGGCGGCCTGATCCTGCCCAACCTCGCC', 'TATCGGCACACGGCGGGGTCGACGCCGA', 'CTCCCCAGCTACGTGGGATCCGAAGCGTCG', 'AGCTCCTGGCACGCCTCGACCTCGTGGC', 'TGACGATCGATAATGCGGCGGATCTCTGTT', 'TATCGGGTCTGAGCTAGTTAGTAGTGCGCCGACCCCCGCCTCGTGCGGATGAGCAATGTCCGAGCACGTATGCAGGCGAGATGATCCATCTTCAAGCCGTATCCGATGCCAAAGTGTGCCCAGGAGGAGATATACTGGGCCAGATCTACG', 'GCTATTCTTGGCAATTCACTTCCAATGAG', 'ACCGCTCGCGCACTCTGGGCCCAGAACGC', 'ACCAACGGCTGGTTCTCGGACGAGACCTACA', 'TAGAGTTCGGCCCCGCCCGATCAGAGCGCGTC', 'GGGGAAGCGTCTATGGCCGACCCGGCCTAA', 'ACTTGTCCCTTCGAGCGGGCTTGTCCGGGCAAGACGGGACTTCGGTCATGCGTGCAGACAATGACACTTTTTCACAGTGTCTCAGTATTGA'],
        'sequence': 'GGCTCTCCCCCGCGCACGCGCGGGTACACCGTCGTGGACGTCTGTCGGCGTACCCTGCAGGATCTCTCCCCCGCGCACGCGCGGGTACACCCTGCGACGAGTTCGAGCGCGGCAGCGCCCCATTTTCTTCCCCGCGCGCGGGTGAACCGGTCGAGATCATACGCTCCGCCCATCCGTCCATCTGTCCTCCCCGCGCGCGCGGGTGAACCGATGTTCACCAAGGATGTCAGCCTGACCCAGCTCTGTCTTCCCCGCGCGCGGGTGAACCCACTCCCTCGCGATGTTTGAGCACGGGTGTTATCTGTCTCCTCCGCGCACGCGCGGGTGAACCGACCGGCGGCCTGATCCTGCCCAACCTCGCCCTGTCTCCTCCGCGCACGCGCGGGTGAACCGTATCGGCACACGGCGGGGTCGACGCCGATCTCCTATATTCCCCGCGCGCGGGTGAACCGCTCCCCAGCTACGTGGGATCCGAAGCGTCGGACTGTCGTCCCCGTGTGCGCGGGTAACCCGAGCTCCTGGCACGCCTCGACCTCGTGGCGGATCTGTCTTCCCCGCGCGCGGGTGAACCATGACGATCGATAATGCGGCGGATCTCTGTTGCCTGTCTTCCCCGCGTGCGCGGGTGATCCATATCGGGTCTGAGCTAGTTAGTAGTGCGCCGACCCCCGCCTCGTGCGGATGAGCAATGTCCGAGCACGTATGCAGGCGAGATGATCCATCTTCAAGCCGTATCCGATGCCAAAGTGTGCCCAGGAGGAGATATACTGGGCCAGATCTACGACTCTCGCACCGAGCACGCGCGGGTAAGCCCGCTATTCTTGGCAATTCACTTCCAATGAGAGCCCTCTCTCCCGCGCACGCGGGTAAACCCACCGCTCGCGCACTCTGGGCCCAGAACGCAATCGCTCCCCGCTCACATGCGGGTAAACCCACCAACGGCTGGTTCTCGGACGAGACCTACAACTCTTCCTCCGCAAGCATGCGGGTGAGCCTTAGAGTTCGGCCCCGCCCGATCAGAGCGCGTCCTTGCTTCCTGGGCACATGCGGGTGAGCAGGGGAAGCGTCTATGGCCGACCCGGCCTAAGCTTCTCTTCGGCATGTATGTGGGTTGCCAGACTTGTCCCTTCGAGCGGGCTTGTCCGGGCAAGACGGGACTTCGGTCATGCGTGCAGACAATGACACTTTTTCACAGTGTCTCAGTATTGACGAGTCTTCCCGCATACGCGCGGGTAATCCG'
    }])

    def create_labels(list_repeats, list_spacers):
        """
        Create a 1D numpy array of 0s and 1s. 0s represent repeats, 1s represent spacers.
        :param list_repeats:
        :param list_spacers: 
        :return: 1D numpy array of 0s and 1s 
        """
        labels = chain.from_iterable([0] * len(repeat) + [1] * len(spacer)
                                    for repeat, spacer in zip_longest(list_repeats, list_spacers, fillvalue=""))
        return np.fromiter(labels, dtype=int)

    # Create labels
    test_dataset["labels"] = test_dataset.apply(lambda x: create_labels(x["list_repeats"], x["list_spacers"]), axis=1)

    # Convert the dataframe to a Hugging Face dataset
    test_dataset = Dataset.from_pandas(test_dataset)

    # Load the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)
    tokenizer.pad_token = "X"

    def tokenize_data(sample, max_length=8000):
        tokenized = tokenizer(sample['sequence'], padding="max_length", truncation=True, max_length=max_length, return_tensors='pt')
        labels = sample['labels']
        # Pad or truncate orf_labels to match the tokenized sequence length
        if len(labels) < max_length:
            labels += [-100] * (max_length - len(labels))
        else:
            labels = labels[:max_length]

        tokenized['labels'] = (torch.tensor(labels, dtype=torch.float32))

        # Remove batch dimension added by tokenizer
        for key in tokenized:
            tokenized[key] = tokenized[key].squeeze(0)

        return tokenized

    # Tokenize the dataset
    test_dataset = test_dataset.map(tokenize_data, batched=False, num_proc=12)

    # Define data collator for dynamic padding within batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO create better solution for this
    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows, that also doesn't support it.
    model.backbone_model.config.model_type = "mpt"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # required, but not used
        report_to="none",  # Disable reporting
        per_device_eval_batch_size=16
    )

    # Initialize Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    # Evaluate the model on the heterogenous array
    print(f"Evaluating heterogenous array with model: {model_dir}")

    # Evaluate the model on the test set
    metrics = trainer.evaluate()

    # For each metric in the metrics dictionary, print the metric name and value
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("")

    # Print out which model is being evaluated
    print(f"Predicting heterogenous array with model: {model_dir}")

    # Run the samples through the model
    prediction_output = trainer.predict(test_dataset)
    predictions_raw = prediction_output.predictions

    # Converts the list of labels to a numpy arrays
    test_labels = np.array(test_dataset["labels"])
    
    # Remove predictions and labels that are padded
    predictions_raw = predictions_raw[test_labels != -100]
    test_labels = test_labels[test_labels != -100]

    # Make predictions to 1 and 0
    predictions = (predictions_raw > 0.5).astype(int)

    # Convert predictions and labels to strings of 1s and 0s for output
    prediction_strings = ["".join(str(pred) for pred in predictions)]
    # Remove all '[' and ']' from the prediction strings
    prediction_strings = [x.replace("[", "").replace("]", "") for x in prediction_strings]
    
    labels_strings = ["".join(str(label) for label in test_labels)]
    # Remove all '[' and ']' from the label strings
    labels_strings = [x.replace("[", "").replace("]", "") for x in labels_strings]

    # Save sequences, labels, and predictions to a file
    output_path = "/work/dlclarge2/koeksalr-crispr/results/predict_heterogenous_array"
    
    if model_id:
        output_path += f"-{model_id}"
    
    output_path += ".txt"

    with open(output_path, "w") as file:
        file.write(f"Sequence:\t\t\t{test_dataset['sequence'][0]}\n")
        file.write(f"Labels:\t\t\t\t{labels_strings[0]}\n")
        file.write(f"Predictions:\t{prediction_strings[0]}\n\n")


    print(f"Results saved to {output_path}")

    
