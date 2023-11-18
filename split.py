import os
import numpy as np
import shutil
root_dir = r"C:/Users/90553/Desktop/BBM467/Proje/Data/"
classes_dir = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]

test_ratio = 0.20


# https://stackoverflow.com/questions/57394135/split-image-dataset-into-train-test-datasets
# here I am doing my train-test split
def split(class_name):
    os.makedirs('C:/Users/90553/Desktop/BBM467/Proje/train/' + class_name)
    os.makedirs('C:/Users/90553/Desktop/BBM467/Proje/test/' + class_name)
    src = root_dir + class_name
    all_file_names = os.listdir(src)
    np.random.shuffle(all_file_names)
    train_file_names, test_file_names = np.split(np.array(all_file_names), [int(len(all_file_names) * (1 - test_ratio))])
    train_file_names = [src + '/' + name for name in train_file_names.tolist()]
    test_file_names = [src + '/' + name for name in test_file_names.tolist()]
    print("*****************************")
    print('Total images: ', len(all_file_names))
    print('Training: ', len(train_file_names))
    print('Testing: ', len(test_file_names))
    print("*****************************")
    for name in train_file_names:
        shutil.copy(name, 'C:/Users/90553/Desktop/BBM467/Proje/train/' + class_name)

    for name in test_file_names:
        shutil.copy(name, 'C:/Users/90553/Desktop/BBM467/Proje/test/' + class_name)


for a in classes_dir:
    split(a)
print("Copying Done!")

