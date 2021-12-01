import os
import glob
import cv2

directory = r'D:\seg\segmentation_models-master\binary_lane_bdd\Labels'
new_directory = r'D:\seg\segmentation_models-master\binary_lane_bdd\Labels_processed'
os.chdir(directory)
for file in glob.glob("*.jpg"):
    img = cv2.imread(file)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 10 < img[i, j, 1] > 0:
                img[i, j, 0], img[i, j, 1], img[i, j, 2] = 255, 255, 255
            else:
                img[i, j, 0], img[i, j, 1], img[i, j, 2] = 0, 0, 0
    cv2.imwrite(os.path.join(new_directory, file), img)
print("preprocessing sucessfull")
