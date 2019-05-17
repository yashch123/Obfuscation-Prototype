from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# get weights for first layer
weights = model.get_weights()[0]

print("Weights: ", weights)

# get the kernal values based on the weights
def get_kernel(weights):
    weight_matrix = np.asmatrix(weights)
    Z = null_space(weights)
    return Z

kernel = get_kernel(weights.transpose())

# create kernel matrix
kernel_matrix = np.asmatrix(kernel)

# obfuscation done based on kernel matrix
def obfuscate(image, kernel_matrix, coefficients):
    image_matrix = image.flatten()
    obfuscation = kernel_matrix.dot(coefficients)
    new_image_matrix = image_matrix + obfuscation
    return np.reshape(new_image_matrix, (28,28))

num_different = 0

for index in range(len(x_test)):
    obfuscated_image = obfuscate(x_test[index], kernel_matrix, [index] * 272)

    # print pictures
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(x_test[index], cmap=plt.cm.binary)
    #plt.xlabel(class_names[y_test[index]])
    #plt.show()

    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(obfuscated_image, cmap=plt.cm.binary)
    #plt.xlabel(class_names[y_test[index]])
    #plt.show()

    # add dimension to single pictures to use model.predict
    og_image_to_run = np.expand_dims(x_test[index],0)
    obfuscated_image_to_run = np.expand_dims(obfuscated_image,0)
    og_image_class = model.predict(og_image_to_run)
    obfuscated_image_class = model.predict(obfuscated_image_to_run)
    # np.argmax returns the highest likelihood class
    if np.argmax(og_image_class) != np.argmax(obfuscated_image_class):
        num_different = num_different + 1
print(num_different)
