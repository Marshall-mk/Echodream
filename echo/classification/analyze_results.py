import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

"""
python echo/classification/analyze_results.py 
    --results_file ./experiments/classification/real/cls_real/test_results.npz 
    --output_file ./experiments/classification/real/cls_real/confusion_matrix.png
"""

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze classification results.")
    parser.add_argument('--results_file', type=str, required=True, help="Path to the .npz results file.")
    parser.add_argument('--output_file', type=str, default='confusion_matrix.png', help="Path to save the confusion matrix image.")
    args = parser.parse_args()

    # Load the test results
    results = np.load(args.results_file)

    # Access the data using dictionary-style notation
    predictions = results['predictions']
    targets = results['targets']
    accuracy = results['accuracy']

    # Display basic information
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # If predictions are probabilities, convert to class labels
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions

    # Generate confusion matrix
    cm = confusion_matrix(targets, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(targets, pred_labels))

    # Visualize confusion matrix with numbers in each cell
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    num_classes = cm.shape[0]
    
    # Add text annotations to show count in each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(args.output_file)
    print(f"Confusion matrix saved as '{args.output_file}'")

if __name__ == "__main__":
    main()
