# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def train(dataset):
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:4]
    Y = dataset[:, 4]
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(Y.reshape(-1, 1))
    #splits
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20)
    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
    model.add(Dense(10, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))
    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=151)
    # evaluate the model
    scores = model.evaluate(test_x, test_y)
    print(model.metrics_names)
    text_out = {
        "acc:": scores[1],
        "loss": scores[0],
    }   
    
    # Saving model in a given location (provided as an env. variable
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.h5")
        model.save(file_path)
    else:
        model.save("model.h5")
        return json.dumps({'message': 'The model was saved locally.'},
                          sort_keys=False, indent=4), 200

    print("Saved the model to disk")
    return json.dumps(text_out, sort_keys=False, indent=4)
