import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Flatten the input data
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
model = keras.Sequential([
 layers.Dense(256, activation="relu", input_shape=(784,)),
 layers.Dense(128, activation="relu"),
 layers.Dense(10, activation="softmax")
])
# Define the optimizers to compare
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
rmsprop_optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
adam_optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Compile the model with each optimizer and train it
model.compile(optimizer=sgd_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
history_sgd = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
model.compile(optimizer=rmsprop_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
history_rmsprop = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
model.compile(optimizer=adam_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
history_adam = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model with each optimizer on the test set
test_loss_sgd, test_acc_sgd = model.evaluate(x_test, y_test)
test_loss_rmsprop, test_acc_rmsprop = model.evaluate(x_test, y_test)
test_loss_adam, test_acc_adam = model.evaluate(x_test, y_test)

print("SGD test accuracy:", test_acc_sgd)
print("RMSprop test accuracy:", test_acc_rmsprop)
print("Adam test accuracy:", test_acc_adam)

