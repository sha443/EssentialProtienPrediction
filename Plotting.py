import os
import pandas as pd
from Utility import Utility
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  # Import necessary functions

# List of feature files
feature_files = ['data/features/biogrid_features.csv', 'data/features/mips_features.csv', 'data/features/dip_features.csv']

# Initialize the utility
util = Utility()

# Prepare dictionaries to hold the results
histories = {}
test_results = {}

# Loop through each feature file
for feature_file_name in feature_files:
    # Check if the file exists
    if not os.path.exists(feature_file_name):
        raise FileNotFoundError(f"File {feature_file_name} does not exist. \n✔ Please run FeatureBuilder.py first")
    else:
        print("Found feature file:", feature_file_name)

    # Load data
    X_train_normalized, X_val_normalized, X_test_normalized, y_train, y_val, y_test = util.load_data(feature_file_name)

    # Initialize the neural network
    nn = NeuralNetwork(input_dim=X_train_normalized.shape[1])

    # Train the model
    history = nn.train(X_train_normalized, y_train, X_val_normalized, y_val, epochs=10)

    # Evaluate the model
    loss, accuracy = nn.evaluate(X_test_normalized, y_test)

    # Store results
    feature_name = os.path.basename(feature_file_name).replace('_features.csv', '')
    histories[feature_name] = history
    test_results[feature_name] = (X_test_normalized, y_test, nn.model)

# Plot accuracy and loss
plt.figure(figsize=(14, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
for feature_name, history in histories.items():
    plt.plot(history.history['loss'], label=f'Train Loss ({feature_name})')
    plt.plot(history.history['val_loss'], label=f'Val Loss ({feature_name})')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
for feature_name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'Train Acc ({feature_name})')
    plt.plot(history.history['val_accuracy'], label=f'Val Acc ({feature_name})')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Plot ROC curve
plt.figure()

for feature_name, (X_test_normalized, y_test, model) in test_results.items():
    y_pred_prob = model.predict(X_test_normalized).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve ({feature_name}, area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
