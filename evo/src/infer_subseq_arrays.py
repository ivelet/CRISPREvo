import sys
from datasets import load_from_disk
import torch
from model import EvoMultiClassTokenClassificationModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
from eval_metrics import compute_metrics
from pathlib import Path
import os

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    # If argument is provided, set the model id
    # Otherwise, use the default model
    model_id = None
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    # If the second argument is provided, set the number of samples
    # Otherwise, use the default amount of samples
    num_samples = 50
    if len(sys.argv) > 2: 
        num_samples = int(sys.argv[2])

    # Load the saved model
    model_dir = f"{ROOT_DIR}/models/crispr_evo"
    if model_id is not None:
        model_dir += f"-{model_id}"

    model = EvoMultiClassTokenClassificationModel.from_pretrained(model_dir)

    data_dir = f"{ROOT_DIR}/data/bona_fide/test"

    print(f"Loading data from {data_dir}")

    # Import the dataset
    test_dataset = load_from_disk(data_dir)

    # Only take the first num_samples samples
    test_dataset = test_dataset.select(range(num_samples))

    # From each sequence and labels extract a subsequence of length 150
    def extract_subsequence(row):
        sequence = row["sequence"]
        labels = row["labels"]

        # Find the first non-2 label
        start = 0
        while labels[start] == 2:
            start += 1

        # Find the last 0 label
        end = len(labels) - 1
        while labels[end] != 0:
            end -= 1
        
        # Select a subsequence of length 150 between the start and end positions
        # By selecting a random number between the start and (end - 150) positions
        # This is done to ensure that the subsequence is not too close to the start or end of the sequence
        start = torch.randint(start, end - 150 + 1, (1,)).item()
        end = start + 150

        return {
            "sequence": sequence[start:end],
            "labels": labels[start:end],
            "start": row["start"] + start,
            "end": row["start"] + end,
        }
    test_dataset = test_dataset.map(extract_subsequence)

    # Load the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)
    tokenizer.pad_token = "X"

    # Tokenize the dataset
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
    test_dataset = test_dataset.map(tokenize_data)

    # Define data collator for dynamic padding within batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows
    model.backbone_model.config.model_type = "mpt"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"{ROOT_DIR}/results",  # required, but not used
        report_to="none",  # Disable reporting
        per_device_eval_batch_size=1
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

    # Print out which model is being evaluated
    print(f"Evaluating model: {model_dir}")

    # Evaluate the model on the test set
    metrics = trainer.evaluate()

    # Save results to a file
    output_path = f"{ROOT_DIR}/results/infer_subseq_arrays"
    os.makedirs(output_path, exist_ok=True)
    output_path += ".txt"

    with open(output_path, "w") as file:
      # For each metric in the metrics dictionary, write the metric name and value
      for metric, value in metrics.items():
          file.write(f"{metric}: {value}\n")
      file.write("\n")

    print(f"Predicting {num_samples} sample(s) from the test set")

    # Run the samples through the model
    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions  # Remove squeeze to preserve class dimension

    test_labels = np.array(test_dataset["labels"])

    prediction_strings = []
    labels_strings = []
    for i in range(num_samples):
        # Create mask for valid tokens
        mask = test_labels[i] != -100
        
        # Apply mask to predictions and labels
        masked_logits = logits[i][mask]  # Shape: (valid_tokens, num_classes)
        masked_labels = test_labels[i][mask]
        
        # Get class predictions using argmax
        probabilities = torch.softmax(torch.from_numpy(masked_logits), dim=-1).numpy()
        predictions = np.argmax(probabilities, axis=-1)

        # Convert to string representation
        prediction_str = ["".join(map(str, predictions))]
        label_str = ["".join(map(str, masked_labels))]
        
        # Clean formatting
        prediction_strings.append([s.replace("[", "").replace("]", "") for s in prediction_str])
        labels_strings.append([s.replace("[", "").replace("]", "") for s in label_str])

    with open(output_path, "a") as file:
        for i in range(num_samples):
            file.write(f"Accession number:\t{test_dataset["acc_number"][i]}\n")
            file.write(f"Start position:\t\t{test_dataset["start"][i]}\n")
            file.write(f"End position:\t\t\t{test_dataset["end"][i]}\n")
            file.write(f"Sequence:\t\t\t\t\t{test_dataset["sequence"][i]}\n")
            file.write(f"Labels:\t\t\t\t\t\t{labels_strings[i][0]}\n")
            file.write(f"Predictions:\t\t\t{prediction_strings[i][0]}\n\n")


    print(f"Results saved to {output_path}")

    
