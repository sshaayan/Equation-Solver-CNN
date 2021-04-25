# Uses the extracted features from the dataset to train a CNN model
# Input: extractedData.csv
# Output: json file for the CNN model

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# ----- PREPARE TRAINING SET -----
trainDF = pd.read_csv("extractedData.csv", index_col = False)

# Convert the labels to binary class matrices
labels = trainDF[["784"]]
trainDF.drop(trainDF.columns[[784]], axis = 1, inplace = True)
labels = np.array(labels)
binLabels = to_categorical(labels, num_classes = 13)

# Reshape the rest of the dataset
finalTrain = []
for i, row in trainDF.iterrows():
	finalTrain.append(np.array(row).reshape(1, 28, 28))


# ----- TRAIN MODEL -----
# Initialize the model
model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape = (1, 28, 28), activation = "relu", 
					padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
model.add(Conv2D(15, (3, 3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(13, activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", 
				metrics = ["accuracy"])

# Train the model on the dataset
model.fit(np.array(finalTrain), binLabels, epochs = 15, batch_size = 300, 
					shuffle = True, verbose = 1)


# ----- OUTPUT -----
outModel = model.to_json()
with open("model.json", "w") as file:
	file.write(outModel)
model.save_weights("model_weights.h5")
