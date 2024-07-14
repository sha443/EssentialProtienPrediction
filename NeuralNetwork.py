from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


class NeuralNetwork:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
    # end def

    def _build_model(self):
        model = Sequential()
        model.add(Dense(1024, bias_initializer="zeros",
                  input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(), metrics=['accuracy'])
        return model
    # end def

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 batch_size=batch_size, validation_data=(X_val, y_val))
        return history
    # end def

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
        return loss, accuracy
    # end def
