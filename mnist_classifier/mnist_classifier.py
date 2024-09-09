import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
# MNIST is a dataset of 70,000 handwritten digits (28x28 pixels each)
mnist = tf.keras.datasets.mnist

# Load and split the MNIST dataset into training and test sets
# x_train: training images, y_train: training labels
# x_test: test images, y_test: test labels

# mnist.load_data() returns a tuple of two tuples: ((x_train, y_train), (x_test, y_test))
# x_train: training images (60,000 images, each 28x28 pixels)
# y_train: corresponding training labels (60,000 labels)
# x_test: test images (10,000 images, each 28x28 pixels)
# y_test: corresponding test labels (10,000 labels)
# This line unpacks these nested tuples into four separate variables
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print shapes to understand the data structure
# x variables contain the image data, y variables contain the corresponding labels
print("Training data shape:", x_train.shape)  # Expected: (60000, 28, 28) - 60000 images, each 28x28 pixels
print("Training labels shape:", y_train.shape)  # Expected: (60000,) - 60000 labels, one for each image
print("Test data shape:", x_test.shape)  # Expected: (10000, 28, 28) - 10000 images, each 28x28 pixels
print("Test labels shape:", y_test.shape)  # Expected: (10000,) - 10000 labels, one for each image
# Print the first 3 labels from the training set
print("First 3 training labels:", y_train[:3])

# Print the first 3 labels from the test set
print("First 3 test labels:", y_test[:3])

# Step 1: Data normalization
# What: Normalize the pixel values of the input images from the range [0, 255] to [0, 1].
# Why: Normalization helps in faster convergence during training and ensures that all input features are on a similar scale,
#      which can improve the performance and stability of the neural network.
# Normalize the pixel values of the training images
# This operation is applied to the entire x_train dataset at once (not recursively)
# It converts the data type to float32 and scales all pixel values from [0, 255] to [0, 1]
# The operation modifies x_train in-place, updating all 60,000 images simultaneously
# The normalization operation updates all 60,000 images simultaneously through vectorized operations.
# NumPy's array operations are designed to work efficiently on entire arrays at once.
# When we divide x_train by 255.0, NumPy applies this operation element-wise to the entire array
# without the need for explicit loops. This is known as broadcasting.

# Here's a breakdown of what happens:
# 1. x_train.astype(np.float32) converts the entire array to float32 data type.
# 2. The division operation (/ 255.0) is then applied to every element in the array.
# 3. These operations leverage optimized, low-level implementations that can utilize
#    parallel processing capabilities of the CPU or GPU, making them very efficient.

# This vectorized approach is much faster than iterating over each image individually,
# as it minimizes Python-level loop overhead and takes advantage of hardware-level optimizations.

# Explanation of normalization in mathematical terms:

# In linear algebra, we can represent the normalization process as follows:

# Let X be the input matrix of shape (n, 28, 28), where n is the number of images.
# Each element x_ijk in X represents a pixel value, where:
# i = image index (1 to n)
# j, k = pixel coordinates (1 to 28 each)

# The normalization operation can be expressed as:

# X_normalized = (1/255) * X

# This is equivalent to applying the following operation to each element:
# x_ijk_normalized = x_ijk / 255

# In matrix notation, this can be written as:
# X_normalized = X ⊙ (1/255)
# where ⊙ represents element-wise multiplication (Hadamard product)

# The result is a new matrix X_normalized with the same shape as X,
# but with all values scaled to the range [0, 1].

# In terms of vector spaces, this operation can be seen as a uniform scaling
# of the input vector space. If we consider each image as a point in a 
# 784-dimensional space (28 * 28 = 784), the normalization scales this space
# uniformly along all dimensions, effectively shrinking it to fit within
# a 784-dimensional unit hypercube.

# This scaling preserves the relative distances between points (images) 
# in the input space, but changes the absolute distances. The benefit is 
# that it brings all features (pixel values) to a common scale, which can 
# help in the optimization process during training.


x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Step 2: Model Architecture
# What: Define the input layer for the neural network model.
# Why: The input layer specifies the shape of the input data (28x28 pixels) that the model will accept,
#      which corresponds to the dimensions of each MNIST digit image.
input_layer = tf.keras.Input(shape=(28, 28))

# Step 3: Flatten the input
# What: Convert the 2D input (28x28) into a 1D array of 784 elements.
# Why: Flattening is necessary to transform the 2D image data into a format that can be fed into a dense neural network layer.
#      This step doesn't affect the data itself but reshapes it for processing by subsequent layers.
flattened = tf.keras.layers.Flatten()(input_layer)

# Step 4: Dense Layers
# What: Add a dense (fully connected) layer with 128 neurons and apply ReLU activation.
# Why: Dense layers learn complex patterns in the data. The ReLU activation introduces non-linearity,
#      allowing the network to learn more sophisticated features and relationships in the input data.
dense1 = tf.keras.layers.Dense(128)(flattened)
dense1_activated = tf.keras.activations.relu(dense1)

# Step 5: Dropout Layer 
# What: Apply a dropout layer that randomly sets 20% of the input units to 0 at each update during training.
# Why: Dropout helps prevent overfitting by reducing interdependent learning between neurons. This improves
#      the model's ability to generalize to new, unseen data.
dropout = tf.keras.layers.Dropout(0.2)(dense1_activated)

# Step 6: Output Layer  
# What: Add the final dense layer with 10 neurons (one for each digit) and apply softmax activation.
# Why: The output layer produces the model's predictions. Softmax activation ensures the output is a probability
#      distribution over the 10 possible digit classes, with values summing to 1.
output_layer = tf.keras.layers.Dense(10)(dropout)
output_activated = tf.keras.activations.softmax(output_layer)

# Step 7: Create the model
# What: Define the complete model by specifying its inputs and outputs.
# Why: This step creates a cohesive model object that encapsulates all the layers and can be used for training,
#      evaluation, and making predictions.
model = tf.keras.Model(inputs=input_layer, outputs=output_activated)

# Step 8: Define the optimizer
# What: Specify the Adam optimizer for training the model.
# Why: Adam is an adaptive learning rate optimization algorithm that combines the benefits of AdaGrad and RMSProp.
#      It's efficient and widely used for training deep neural networks.
optimizer = tf.keras.optimizers.Adam()

# Step 9: Define the loss function
# What: Specify Sparse Categorical Crossentropy as the loss function.
# Why: This loss function is appropriate for multi-class classification problems where the classes are mutually exclusive
#      (each input belongs to only one class) and the labels are integers.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Step 10: Compile the model
# What: Compile the model with the specified optimizer, loss function, and metrics.
# Why: Compilation configures the model for training by specifying how it should be optimized (optimizer),
#      how it should measure its error (loss function), and what metrics to track during training and testing.
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Step 11: Train the model
# What: Fit the model to the training data for 5 epochs, using 10% of the data for validation.
# Why: This step performs the actual training of the model, adjusting its weights to minimize the loss function.
#      The validation split allows monitoring of the model's performance on unseen data during training.
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)

# Step 12: Evaluate the model on the test set
# What: Assess the model's performance on the test dataset.
# Why: Evaluation on a separate test set provides an unbiased estimate of the model's performance on new, unseen data,
#      which is crucial for understanding how well the model generalizes.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_accuracy:.4f}')

# Step 13: Select a single test image
# What: Choose a single image from the test set for prediction demonstration.
# Why: This allows for a concrete, visual example of how the model performs on individual instances,
#      which can be helpful for understanding and presenting the model's capabilities.
test_image_index = 0
test_image = x_test[test_image_index]
true_label = y_test[test_image_index]

# Step 14: Make a prediction on the single test image
# What: Use the trained model to predict the digit in the selected test image.
# Why: This demonstrates how to use the model for inference on new data and allows for a comparison
#      between the model's prediction and the true label.
prediction = model.predict(test_image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

# Step 15: Display the test image
# What: Visualize the selected test image along with its true and predicted labels.
# Why: This provides a visual confirmation of the model's performance, allowing for an intuitive understanding
#      of how well the model is doing on individual examples.
plt.figure(figsize=(5,5))
plt.imshow(test_image, cmap='gray')
plt.title(f"True Label: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()

# Step 16: Print the true and predicted labels
# What: Output the true label, predicted label, and the confidence of the prediction.
# Why: This provides a quantitative summary of the model's performance on the specific example,
#      including how confident the model is in its prediction.
print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Confidence: {np.max(prediction):.4f}")