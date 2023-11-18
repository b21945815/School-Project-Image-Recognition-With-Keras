from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import RMSprop
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import KFold

IMG_SIZE = 330


# https://stackoverflow.com/questions/65179822/how-to-convert-rgb-images-to-grayscale-expand-dimensions-of-that-grayscale-imag
def to_grayscale_then_rgb(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image, label


# https://stackoverflow.com/questions/63572451/normalizing-batchdataset-in-tensorflow-2-3
def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label


def process_x(image, label):
    return image


def process_y(image, label):
    return label


# For transfer learning
# https://wngaw.github.io/transfer-learning-for-image-classification/#transfer-learning-using-inception-v3
# https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b

# for k-fold
# https://stackoverflow.com/questions/71676222/how-apply-kfold-cross-validation-using-tf-keras-utils-image-dataset-from-directo
train_ds = tf.keras.utils.image_dataset_from_directory("train", validation_split=0.25,
                                                       subset="training", seed=123, image_size=(IMG_SIZE, IMG_SIZE), batch_size=100, shuffle=False)

validation_data = tf.keras.utils.image_dataset_from_directory("train", validation_split=0.25,
                                                              subset="validation", seed=123, image_size=(IMG_SIZE, IMG_SIZE), batch_size=100, shuffle=False)

train_ds = train_ds.map(process)
validation_data = validation_data.map(process)

train_ds = train_ds.map(to_grayscale_then_rgb)
validation_data = validation_data.map(to_grayscale_then_rgb)

inputs = np.concatenate(list(train_ds.map(process_x)))
targets = np.concatenate(list(train_ds.map(process_y)))

val_images = np.concatenate(list(validation_data.map(process_x)))
val_labels = np.concatenate(list(validation_data.map(process_y)))

inputs = np.concatenate((inputs, val_images), axis=0)
targets = np.concatenate((targets, val_labels), axis=0)

k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
Counter = 0
foldAccuracy = []
foldLoss = []
foldValidationAccuracy = []
foldValidationLoss = []

for train, test in k_fold.split(inputs, targets):
    validation_data = tf.data.Dataset.from_tensor_slices((inputs[test], targets[test])).batch(100)
    train_ds = tf.data.Dataset.from_tensor_slices((inputs[train], targets[train])).batch(100)
    pre_trained_model = InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    for layer in pre_trained_model.layers[:209]:
        layer.trainable = False
    layers = tf.keras.layers

    x = pre_trained_model.output
    # Transfer learning is completed

    x = layers.AveragePooling2D(pool_size=(8, 8), strides=1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(8, activation="softmax")(x)

    model = Model(inputs=pre_trained_model.input, outputs=x)
    # print("Number of layers in the base model: ", len(model.layers))

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    checkpoint = ModelCheckpoint("CheckPoint/checkpoint" + str(Counter), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
    epochs = 6
    history = model.fit(train_ds, validation_data=validation_data, epochs=epochs, verbose=1, callbacks=[checkpoint, early])

    foldAccuracy.append(history.history["accuracy"][-1])
    foldLoss.append(history.history["loss"][-1])
    foldValidationAccuracy.append(history.history["val_accuracy"][-1])
    foldValidationLoss.append(history.history["val_loss"][-1])

    # Saving the model
    model_json = model.to_json()
    with open("model" + str(Counter) + ".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model" + str(Counter) + ".h5")
    print("Saved model to disk")

    model.save("Inception" + str(Counter) + ".model2")
    Counter = Counter + 1

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(foldLoss)), foldLoss, label="train_loss")
plt.plot(np.arange(0, len(foldValidationLoss)), foldValidationLoss, label="validation_loss")
plt.plot(np.arange(0, len(foldAccuracy)), foldAccuracy, label="train_accuracy")
plt.plot(np.arange(0, len(foldValidationAccuracy)), foldValidationAccuracy, label="validation_accuracy")
plt.title("5-fold cross validation")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
