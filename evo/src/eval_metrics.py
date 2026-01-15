import evaluate

# Load the accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Get the predicted class by taking the argmax along the last dimension
    predictions = predictions.argmax(-1)
    
    # Define token IDs for 'X' and '!'
    x_token_id = 88
    exclamation_token_id = 33

    # Create a mask to filter out labels that are equal to x_token_id or exclamation_token_id
    mask = (labels != x_token_id) & (labels != exclamation_token_id)
    
    # Filter predictions and labels based on the mask
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]
    
    # Compute and return accuracy only for valid predictions
    return metric.compute(predictions=filtered_predictions, references=filtered_labels)
