from keras_sequential_ascii import keras2ascii
from keras.utils import plot_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

from Utility import Utility
from NeuralNetwork import NeuralNetwork
from imblearn.over_sampling import SMOTE

feature_file_name = 'data/features/biogrid_features.csv'
feature_file_name_test = 'data/features/mips_features.csv'
# feature_file_name = 'data/features/dip_features.csv'

# check if file exists
if not os.path.exists(feature_file_name):
    raise FileNotFoundError(
        f"File {feature_file_name} does not exist. \n✔ Please run FeatureBuilder.py first")
else:
    print("Found feature file: ", feature_file_name)
# end file checkingpip install pydot


# Load data
data = pd.read_csv(feature_file_name)
data_test = pd.read_csv(feature_file_name_test)

# Separate features and labels and Split data into training and testing sets
X = data.drop(['Essentiality', 'Name'], axis=1)
y = data['Essentiality']

X_test = data_test.drop(['Essentiality', 'Name'], axis=1)
y_test = data_test['Essentiality']

# Apply SMOTE
print("Instances Before SMOTE: ", len(y))
smote = SMOTE(random_state=1, sampling_strategy='minority',
              n_jobs=-1, k_neighbors=5)
X, y = smote.fit_resample(X, y)
print("Instances After SMOTE: ", len(y))

# test train split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Normalize features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)


# Initialize the neural network
nn = NeuralNetwork(input_dim=X_train_normalized.shape[1])

# Train the model
history = nn.train(X_train_normalized, y_train,
                   X_val_normalized, y_val, epochs=100)

# Evaluate the model
loss, accuracy = nn.evaluate(X_test_normalized, y_test)

util = Utility()
util.plot_model_performance(nn.model, history, X_test_normalized, y_test)


# plot_model(nn.model, to_file='NN_model.png', show_shapes=True,
#            show_layer_activations=True, show_dtype=False, show_layer_names=True)

# keras2ascii(nn.model)
