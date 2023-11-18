import os
import cv2
import imutils

classes_dir = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]

# The file number of the smallest folder must be at least (file_number/4)
# As a result of these operations, there may be a few extra photos in some folders.
file_number = 3330
root_dir = r"C:/Users/90553/Desktop/BBM467/Proje/Data/"

slist = os.listdir(root_dir)
# https://stackoverflow.com/questions/66986830/image-augmentation-with-tensorflow-so-all-classes-have-exact-same-number-of-imag
# I'm doing the Image Augmentation technique here
rows = 360
cols = 363
for classX in slist:
    class_path = os.path.join(root_dir, classX)
    filelist = os.listdir(class_path)
    file_count = len(filelist)
    delta = file_number-file_count
    i = 0
    k = 0
    while k < delta:
        k = k + 3
        file = filelist[i]
        file_split = os.path.split(file)
        index = file_split[1].rfind('.')
        file_name = file[:index]
        ext = file[index:]
        fpath = os.path.join(class_path, file)
        image = cv2.imread(fpath)

        label = '-new'
        file_new_name = file_name + '-' + str(i) + '-' + label + ext
        Rotated_image = imutils.rotate(image, angle=90)
        destination_path = os.path.join(class_path, file_new_name)
        cv2.imwrite(destination_path, Rotated_image)

        label = '-new2'
        file_new_name = file_name + '-' + str(i) + '-' + label + ext
        Rotated_image = imutils.rotate(image, angle=180)
        destination_path = os.path.join(class_path, file_new_name)
        cv2.imwrite(destination_path, Rotated_image)

        label = '-new3'
        file_new_name = file_name + '-' + str(i) + '-' + label + ext
        Rotated_image = imutils.rotate(image, angle=270)
        destination_path = os.path.join(class_path, file_new_name)
        cv2.imwrite(destination_path, Rotated_image)
        i = i + 1







