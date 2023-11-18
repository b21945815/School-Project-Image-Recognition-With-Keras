import numpy as np
import matplotlib.pyplot as plt

# Teacher, I got the results from terminal because I print the array. The error turned me off from the graphics :(
loss = [0.535, 0.143, 0.101, 0.071, 0.0555, 0.0436, 0.0369, 0.0311, 0.0315, 0.0263, 0.0223, 0.0262, 0.0220]
val_loss = [0.735, 0.618, 0.694, 1.321, 1.4201, 1.0683, 1.4876, 1.2336, 1.6318, 1.9210, 1.4692, 2.0365, 1.8292]
accuracy = [0.866, 0.952, 0.966, 0.976, 0.982, 0.9859, 0.9882, 0.9896, 0.9903, 0.9913, 0.9933, 0.9930, 0.9942]
val_accuracy = [0.827, 0.862, 0.865, 0.848, 0.8069, 0.8689, 0.8355, 0.8797, 0.8430, 0.8663, 0.8581, 0.8588, 0.8630]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 13), loss, label="train_loss")
plt.plot(np.arange(0, 13), val_loss, label="validation_loss")
plt.plot(np.arange(0, 13), accuracy, label="train_accuracy")
plt.plot(np.arange(0, 13), val_accuracy, label="validation_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

y = np.array([901, 3117, 1551, 2895, 1214, 1420, 3329, 2348])
Labels = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]

plt.pie(y, autopct="%1.1f%%", labels=Labels)
plt.show()

