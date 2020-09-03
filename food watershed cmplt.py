import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

files = []


os.chdir(r"C:\Users\abdussamad p\PycharmProjects\pythonProject\venv\food20dataset\train_set")
folders = os.listdir()

for folder in folders:
    parent = os.path.abspath(folder)
    for file1 in os.listdir(folder):
        files.append(os.path.join(parent,file1))

def preprocess(image_file):
    print(image_file)

    img = cv2.imread(image_file)

    width = 220
    height = 220
    dim = (width, height)
# resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#denoise

    dst = cv2.fastNlMeansDenoisingColored(resized,None,10,10,7,21)

    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()


#segmentation


    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    plt.imshow(thresh)
    plt.show()

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    plt.imshow(opening)
    plt.show()

    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    plt.imshow(sure_bg)
    plt.show()

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    plt.imshow(sure_fg)
    plt.show()

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    plt.imshow(unknown)
    plt.show()

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers+1

    markers[unknown==255] = 0

    markers = cv2.watershed(dst,markers)
    dst[markers == -1] = [255,0,0]

    plt.imshow(dst)
    plt.show()

for file in files:
    preprocess(file)
