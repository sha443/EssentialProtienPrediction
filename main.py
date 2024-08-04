from keras_sequential_ascii import keras2ascii
from keras.utils import plot_model
import os

from Utility import Utility
from NeuralNetwork import NeuralNetwork

feature_file_name = 'data/features/biogrid_features_sequence.csv'
# feature_file_name = 'data/features/mips_features.csv'
# feature_file_name = 'data/features/dip_features.csv'

# check if file exists
if not os.path.exists(feature_file_name):
    raise FileNotFoundError(
        f"File {feature_file_name} does not exist. \n✔ Please run FeatureBuilder.py first")
else:
    print("Found feature file: ", feature_file_name)
# end file checkingpip install pydot


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


# plot_model(nn.model, to_file='NN_model.png', show_shapes=True,
#            show_layer_activations=True, show_dtype=False, show_layer_names=True)

# keras2ascii(nn.model)
