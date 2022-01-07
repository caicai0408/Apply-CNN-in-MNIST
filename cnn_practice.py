import numpy as np
from Mnist.mnist import MNIST, label_packer, img_packer
#There's a difference here, My Mnist package in tool library name Mnist, if your package have different name, please import it correctly.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# --------------------------------Preparing data--------------------------------
mndata = MNIST(r'xxxxxxxx', return_type='numpy', gz=True)
#xxxxxxxx is your mnist data file directory
#if your data already be decompressed, gz=Flase, if not, gz=True

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


# Normalize the images.
# we’ll normalize the image pixel values from [0, 255] to [-0.5, 0.5] to make our network easier to train (using smaller, centered values usually leads to better results).
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
# We need reshape each image from (28, 28) to (28, 28, 1) because Keras requires the third dimension.
train_images=train_images.reshape(-1, 28, 28, 1)
test_images=test_images.reshape(-1, 28, 28, 1)


print(train_images.shape)
print(test_images.shape)

# --------------------------------Keras--------------------------------

num_filters = 8
filter_size = 2
pool_size = 2

# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1), activation='relu'),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  validation_data=(test_images, to_categorical(test_labels)),
)

Save the model to disk.
model.save_weights('cnn.mnist')

Load the model from disk later using:
model.load_weights('cnn.mnist')

# Predict on the first 8 test images.
predictions = model.predict(test_images[:10])

# Print our model's predictions.
print('model''s predictions')
print(np.argmax(predictions, axis=1))

# Check our predictions against the ground truths.
print('checker')
print(test_labels[:10])
