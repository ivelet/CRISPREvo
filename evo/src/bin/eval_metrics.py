import sys
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score
import torch
from model import EvoBinaryTokenClassificationModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

def compute_metrics(eval_pred):
    """
    Compute metrics for HuggingFace Trainer.
    
    Args:
        eval_pred (tuple): A tuple of (logits, labels) where:
            - logits: Model predictions (logits before applying any activation).
            - labels: Ground truth labels.

    Returns:
        dict: Dictionary containing computed metrics (accuracy, matthews_corrcoef, precision, recall, F1, ROC AUC).
    """
    # Unpack the logits and labels
    logits, labels = eval_pred # Shapes (batch_size, sequence_length, 1) and (batch_size, sequence_length) respectively

    # Convert to 2D tensor and remove singleton dimension
    logits = logits.squeeze(-1)   # (batch_size, seq_len)

    # Create boolean mask for valid tokens (non-padding)
    mask = labels != -100  # (batch_size, seq_len)

    # Apply mask to flatten only valid tokens
    masked_labels = labels[mask]  # (num_valid_tokens,)
    masked_logits = logits[mask]  # (num_valid_tokens,)
    
    # Parse logits through a sigmoid function
    predictions = torch.sigmoid(torch.from_numpy(masked_logits)).numpy()

    # Create true/false predictions and labels based on the threshold of 0.5
    masked_labels = (masked_labels > 0).astype(int)
    predictions = predictions > 0.5

    # Compute metrics
    accuracy = accuracy_score(masked_labels, predictions)
    mcc = matthews_corrcoef(masked_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(masked_labels, predictions, average="binary")
    roc_auc = roc_auc_score(masked_labels, predictions)

    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


if __name__ == "__main__":
    # If argument is provided, set take the model id from the argument
    # Otherwise, use the default model
    model_id = None
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    # Load the saved model
    model_dir = "/work/dlclarge2/koeksalr-crispr/results/bin/model"
    if model_id is not None:
        model_dir += f"-{model_id}"

    model = EvoBinaryTokenClassificationModel.from_pretrained(model_dir)

    data_dir = "/work/dlclarge2/koeksalr-crispr/datasets/evo/bin_data_only_bona_fide/test"
    print(f"Loading data from {data_dir}")

    # Import the dataset
    test_dataset = load_from_disk(data_dir)

    # Discard the sequence column
    test_dataset = test_dataset.remove_columns("sequence")

    # Load the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)
    tokenizer.pad_token = "X"
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # required, but not used
        report_to="none",  # Disable reporting
        per_device_eval_batch_size=1
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        eval_dataset=test_dataset,
        args=training_args
    )

    # Print out which model is being evaluated
    print(f"Evaluating model: {model_dir}")

    # TODO create better solution for this
    # In order to use PEFT with StripedHyena, that doesn't support input_embeds, we set it to a model type that PEFT knows, that also doesn't support it.
    model.backbone_model.config.model_type = "mpt"

    # Evaluate the model on the test set
    metrics = trainer.evaluate()

    # For each metric in the metrics dictionary, print the metric name and value
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
