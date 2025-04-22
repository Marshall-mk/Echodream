import numpy as np

# Load the data
t = np.load(
    "/home/khmuhammad/Echo-Dream/experiments/classification/real/cls_real/test_results.npz"
)

# Print available keys
print("Available keys:", list(t.keys()))

# Access data correctly
predictions = t["predictions"]
targets = t["targets"]
accuracy = t["accuracy"]

# Print information
print(f"Accuracy: {accuracy}")
print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")

# Print sample data (first 5 entries)
print("\nFirst 5 predictions:", predictions[:5])
print("First 5 targets:", targets[:5])
