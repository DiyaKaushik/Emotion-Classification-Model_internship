from tensorflow import keras
from tensorflow.keras import layers

class EmotionClassificationModel:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Embedding(input_dim=10000, output_dim=128))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)