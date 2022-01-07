import numpy as np
# from Mnist.mnist import MNIST
from Mnist.mnist import MNIST, label_packer, img_packer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# --------------------------------Preparing data--------------------------------
mndata = MNIST(r'/home/huang/pyvenv/lib/python3.6/site-packages/Mnist/mnist', return_type='numpy', gz=True)


train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images=train_images.reshape(-1, 28, 28, 1)
test_images=test_images.reshape(-1, 28, 28, 1)
# Reshape the images.
# train_images = np.expand_dims(train_images, axis=3)
# test_images = np.expand_dims(test_images, axis=3)

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

# Save the model to disk.
# model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 8 test images.
predictions = model.predict(test_images[:10])

# Print our model's predictions.
print('model''s predictions')
print(np.argmax(predictions, axis=1))

# Check our predictions against the ground truths.
print('checker')
print(test_labels[:10])
