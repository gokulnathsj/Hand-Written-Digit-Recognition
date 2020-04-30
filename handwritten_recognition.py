import numpy as np
from mnist import MNIST 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load the dataset
mnist = MNIST('MNIST')
train_images, train_labels = mnist.load_training()
test_images, test_labels = mnist.load_testing() 

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalising the value
train_images = (train_images/255) 
test_images = (test_images/255) 

# Building the model for prediction
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

# Train the model
model.fit(train_images, to_categorical(train_labels), epochs = 10, batch_size = 64)

# Evaluating the model 
model.evaluate(test_images, to_categorical(test_labels))

model.save_weights('model.h5')

# predicting the results
y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)

# Making the Confusion Matrix and accuracy of prediction in the test set
count = 0
for i in range(len(y_pred)):
    if y_pred[i] == test_labels[i]:
        count += 1
accuracy = count * 100 / len(y_pred)
print(accuracy)


# printing the first 5 digits
for i in range(0, 5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype = 'float')
    pixels = first_image.reshape(28, 28)
    plt.imshow(pixels, cmap='gray')
    plt.show()
