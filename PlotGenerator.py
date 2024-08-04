import os
import pandas as pd
from Utility import Utility
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

# List of feature files
feature_files = ['data/features/biogrid_centrality.csv',
                 'data/features/dip_centrality.csv', 'data/features/mips_centrality.csv']

# Mapping of file base names to formatted names
formatted_names = {
    'biogrid': 'BioGRID',
    'dip': 'DIP',
    'mips': 'MIPS'
}

# Initialize the utility
util = Utility()

# Prepare dictionaries to hold the results
histories = {}
test_results = {}
metrics = {}

# Loop through each feature file
for feature_file_name in feature_files:
    # Check if the file exists
    if not os.path.exists(feature_file_name):
        raise FileNotFoundError(
            f"File {feature_file_name} does not exist. \n✔ Please run FeatureBuilder.py first")
    else:
        print("Found feature file:", feature_file_name)

    # Load data
    X_train_normalized, X_val_normalized, X_test_normalized, y_train, y_val, y_test = util.load_data(
        feature_file_name)

    # Initialize the neural network
    nn = NeuralNetwork(input_dim=X_train_normalized.shape[1])

    # Train the model
    history = nn.train(X_train_normalized, y_train,
                       X_val_normalized, y_val, epochs=10)

    # Evaluate the model
    loss, accuracy = nn.evaluate(X_test_normalized, y_test)

    # Store results
    feature_name = os.path.basename(
        feature_file_name).replace('_centrality.csv', '')
    histories[feature_name] = history
    test_results[feature_name] = (X_test_normalized, y_test, nn.model)

    # Calculate precision, recall, F1-score, and accuracy
    y_pred = (nn.model.predict(X_test_normalized) > 0.5).astype("int32")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary')
    metrics[feature_name] = (precision, recall, f1, accuracy)

# Print precision, recall, F1-score, and accuracy for each feature set
for feature_name, (precision, recall, f1, accuracy) in metrics.items():
    formatted_name = formatted_names[feature_name]
    print(f"{formatted_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(14, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
for feature_name, history in histories.items():
    formatted_name = formatted_names[feature_name]
    plt.plot(history.history['loss'], label=f'Train Loss ({formatted_name})')
    plt.plot(history.history['val_loss'],
             label=f'Val Loss ({formatted_name})')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
for feature_name, history in histories.items():
    formatted_name = formatted_names[feature_name]
    plt.plot(history.history['accuracy'],
             label=f'Train Acc ({formatted_name})')
    plt.plot(history.history['val_accuracy'],
             label=f'Val Acc ({formatted_name})')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('output/accuracy_loss_plot.pdf')
plt.close()

# Plot ROC curve
plt.figure()

for feature_name, (X_test_normalized, y_test, model) in test_results.items():
    formatted_name = formatted_names[feature_name]
    y_pred_prob = model.predict(X_test_normalized).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2,
             label=f'ROC ({formatted_name}, AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('output/roc_curve_plot.pdf')
plt.close()

# Plot confusion matrices
for feature_name, (X_test_normalized, y_test, model) in test_results.items():
    formatted_name = formatted_names[feature_name]
    y_pred = (model.predict(X_test_normalized) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({formatted_name})')
    plt.savefig(f'output/confusion_matrix_{feature_name}.pdf')
    plt.close()
