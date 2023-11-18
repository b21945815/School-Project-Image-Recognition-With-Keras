import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("Inception.model")
IMG_SIZE = 330
DATADIR = "C:/Users/90553/Desktop/PBC_dataset_normal_DIB"
Categories = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]


#  In order not to make the estimations difficult due to the tonal differences in the incoming pictures,
def to_grayscale_then_rgb(visual):
    visual = tf.image.rgb_to_grayscale(visual)
    visual = tf.image.grayscale_to_rgb(visual)
    return visual


tr_data = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=to_grayscale_then_rgb)
test_data = tr_data.flow_from_directory(directory="test", target_size=(IMG_SIZE, IMG_SIZE), batch_size=50, class_mode="binary")
# Important
size = 8*3330

# Predictions
predicts = model.predict(test_data, workers=2)
predictions = np.reshape(predicts, (-1, 8))
predictions = np.argmax(predictions, axis=1)
label_index = {v: k for k, v in test_data.class_indices.items()}
predictions = [label_index[p] for p in predictions]


success = 0
label_list = []
counter = 0

for image, label in test_data:
    list2 = label
    for i in range(0, 50):
        counter = counter + 1
        if counter != size + 1:
            label_list.append(Categories[int(list2[i])])
            if Categories[int(list2[i])] == predictions[i]:
                success = success + 1
        else:
            break
    if counter == size + 1:
        break


print(success/len(predictions))
# https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
cm = confusion_matrix(label_list, predictions, labels=Categories)
cm_df = pd.DataFrame(cm, index=Categories, columns=Categories)
plt.figure(figsize=(12, 12))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.figure(figsize=(12, 12))
sns.heatmap(cm_df/np.sum(cm_df), annot=True, fmt='.2%', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
