import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class Utility:
    def __init__(self):
        pass

    def load_data(self, feature_file):
        data = pd.read_csv(feature_file)

        # Separate features and labels and Split data into training and testing sets
        X = data.drop(['Essentiality', 'Name'], axis=1)
        y = data['Essentiality']

        # Apply SMOTE
        print("Instances Before SMOTE: ", len(y))
        smote = SMOTE(random_state=1, sampling_strategy='minority',
                      n_jobs=-1, k_neighbors=5)
        X, y = smote.fit_resample(X, y)
        print("Instances After SMOTE: ", len(y))

        # test train split
        X_seen, X_test, y_seen, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(
            X_seen, y_seen, test_size=0.1, random_state=7)

        # Normalize features
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_val_normalized = scaler.transform(X_val)
        X_test_normalized = scaler.transform(X_test)

        return X_train_normalized, X_val_normalized, X_test_normalized, y_train, y_val, y_test

    def plot_model_performance(self, model, history, X_test_normalized, y_test):

        # Plot training & validation loss values
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'], loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.tight_layout()
        plt.show()

        # Predict on test data
        y_pred = (model.predict(X_test_normalized) > 0.50).astype("int32")

        # Using sklearn's built-in functions
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        # ROC Curve and AUC
        y_pred_prob = model.predict(X_test_normalized).ravel()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

# end function
# end class
