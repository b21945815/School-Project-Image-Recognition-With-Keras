from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model


IMG_SIZE = 330


# https://stackoverflow.com/questions/65179822/how-to-convert-rgb-images-to-grayscale-expand-dimensions-of-that-grayscale-imag
def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image


# For transfer learning
# https://wngaw.github.io/transfer-learning-for-image-classification/#transfer-learning-using-inception-v3
# https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b
# 170(100,25,0.0011) 128(84,21,0.0013) Some of the trials
tr_data = ImageDataGenerator(rescale=1.0/255.0, fill_mode="nearest", preprocessing_function=to_grayscale_then_rgb, validation_split=0.2)
data = tr_data.flow_from_directory(directory="train", subset="training", seed=123, shuffle=True, target_size=(IMG_SIZE, IMG_SIZE), batch_size=100, class_mode="binary")
validation = tr_data.flow_from_directory(directory="train", subset="validation", seed=123, shuffle=True, target_size=(IMG_SIZE, IMG_SIZE), batch_size=100, class_mode="binary")
pre_trained_model = InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
# [:289]: [:209]: 299 Some of the trials
for layer in pre_trained_model.layers[:289]:
    layer.trainable = False
layers = tf.keras.layers

x = pre_trained_model.output
# Transfer learning is completed
x = layers.AveragePooling2D(pool_size=(8, 8), strides=1)(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dense(8, activation="softmax")(x)

model = Model(inputs=pre_trained_model.input, outputs=x)

# model.summary()
# print(len(model.layers))
model.compile(optimizer=RMSprop(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
checkpoint = ModelCheckpoint("CheckPoint/checkpoint", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
# epochs = 100
epochs = 100
history = model.fit(data, validation_data=validation, epochs=epochs, verbose=1, callbacks=[checkpoint, early])

# Graph for evaluation

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(history.history["loss"])), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(history.history["val_loss"])), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(history.history["accuracy"])), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(history.history["val_accuracy"])), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('Inception.model')
