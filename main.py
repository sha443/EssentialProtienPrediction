import os

from Utility import Utility
from NeuralNetwork import NeuralNetwork

feature_file_name = 'data/features/biogrid_features.csv'
# feature_file_name = 'data/features/dip_features.csv'

# check if file exists
if not os.path.exists(feature_file_name):
    raise FileNotFoundError(
        f"File {feature_file_name} does not exist. \n✔ Please run FeatureBuilder.py first")
else:
    print("Found feature file: ", feature_file_name)
# end file checking


# Load data
util = Utility()
X_train_normalized, X_val_normalized, X_test_normalized, y_train, y_val, y_test = util.load_data(
    feature_file_name)


# Initialize the neural network
nn = NeuralNetwork(input_dim=X_train_normalized.shape[1])

# Train the model
history = nn.train(X_train_normalized, y_train,
                   X_val_normalized, y_val, epochs=10)

# Evaluate the model
loss, accuracy = nn.evaluate(X_test_normalized, y_test)

util.plot_model_performance(nn.model, history, X_test_normalized, y_test)
