import sys
from datasets import load_from_disk
from model import EvoBinaryTokenClassificationModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
from eval_metrics import compute_metrics

if __name__ == "__main__":
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
    model_dir = "/work/dlclarge2/koeksalr-crispr/results/bin/model"
    if model_id is not None:
        model_dir += f"-{model_id}"

    model = EvoBinaryTokenClassificationModel.from_pretrained(model_dir)

    data_dir = "/work/dlclarge2/koeksalr-crispr/datasets/evo/bin_data_only_bona_fide/test"

    print(f"Loading data from {data_dir}")

    # Import the dataset
    test_dataset = load_from_disk(data_dir)

    # Load the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)
    tokenizer.pad_token = "X"

    # Only take the first num_samples samples
    test_dataset = test_dataset.select(range(num_samples))

    # Tokenize the dataset
    test_dataset = test_dataset.map(
        lambda examples: tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=8000),
        batched=True
    )

    # Define data collator for dynamic padding within batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO create better solution for this
    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows, that also doesn't support it.
    model.backbone_model.config.model_type = "mpt"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # required, but not used
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

    # For each metric in the metrics dictionary, print the metric name and value
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("")

    print(f"Predicting {num_samples} sample(s) from the test set")

    # Run the samples through the model
    prediction_output = trainer.predict(test_dataset)
    predictions_raw = prediction_output.predictions.squeeze(-1)

    # Converts the list of labels to a numpy arrays
    test_labels = np.array(test_dataset["labels"])

    prediction_strings = []
    labels_strings = []
    for i in range(num_samples):     
        # Remove predictions and labels that are padded
        masked_predictions = predictions_raw[i][test_labels[i] != -100]
        masked_labels = test_labels[i][test_labels[i] != -100]
        # Make predictions to 1 and 0
        predictions = (masked_predictions > 0.5).astype(int)

        # Convert predictions and labels to strings of 1s and 0s for output
        prediction_string = ["".join(str(pred) for pred in predictions)]
        # Remove all '[' and ']' from the prediction strings
        prediction_string = [x.replace("[", "").replace("]", "") for x in prediction_string]

        prediction_strings.append(prediction_string)
        
        labels_string = ["".join(str(label) for label in masked_labels)]
        # Remove all '[' and ']' from the label strings
        labels_string = [x.replace("[", "").replace("]", "") for x in labels_string]

        labels_strings.append(labels_string)

    # Save sequences, labels, and predictions to a file
    output_path = "/work/dlclarge2/koeksalr-crispr/results/infer"
    
    if model_id:
        output_path += f"-{model_id}"
    
    output_path += ".txt"

    with open(output_path, "w") as file:
        for i in range(num_samples):
            file.write(f"Accession number:\t{test_dataset["acc_number"][i]}\n")
            file.write(f"Start position:\t\t{test_dataset["start"][i]}\n")
            file.write(f"End position:\t\t\t{test_dataset["end"][i]}\n")
            file.write(f"Sequence:\t\t\t\t\t{test_dataset["sequence"][i]}\n")
            file.write(f"Labels:\t\t\t\t\t\t{labels_strings[i][0]}\n")
            file.write(f"Predictions:\t\t\t{prediction_strings[i][0]}\n\n")


    print(f"Results saved to {output_path}")

    
