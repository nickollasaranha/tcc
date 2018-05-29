import cv2
import numpy as np
import os, sys
import re
from os import listdir

import Preprocess

train_directory = "C:\\Users\\Nickollas Aranha\\Documents\\tcc\\database\\training\\"
test_directory = "C:\\Users\\Nickollas Aranha\\Documents\\tcc\\database\\testing\\"
text_extension = ".txt"
image_extension = ".png"

## HOG Vars
img_shape = [24, 16]
cell_size = (4, 4)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

def get_char(char, imgThreshScene, dataset):
    char_centerX = int(char[0])
    char_centerY = int(char[1])
    char_width = int(char[2])
    char_height = int(char[3])
    imgChar = imgThreshScene[char_centerY:char_centerY+char_height, char_centerX:char_centerX+char_width]
    imgChar = cv2.resize(imgChar, (img_shape[1], img_shape[0]))
    #cv2.imshow("try", imgChar)
    #cv2.waitKey(0)
    dataset.append(imgChar)

def get_track_info(track, dataset, responses):
    # Open .txt file and image
    file = open(track+text_extension)
    imgOriginalScene = cv2.imread(track+image_extension)
    imgThreshScene, _ = Preprocess.preprocess(imgOriginalScene)

    # Get all chars position
    lines = file.readlines()
    chars_position = [re.sub("[a-z:\n]", "", line).split(" ")[2:] for line in lines[8:]]
    list_char = [get_char(char.copy(), imgThreshScene, dataset) for char in chars_position]

    # Get plate number
    plate_number = list(re.sub("[-\nplate: ]", "", lines[6]))
    for char in plate_number:
        responses.append(ord(char))

def createDataset(directory):
    dataset, responses = [], []
    for folder in listdir(directory):
        tracks = [os.path.splitext(directory+folder+"\\"+track)[0] for track in listdir(directory+folder) if os.path.splitext(directory+folder+"\\"+track)[1]==".txt"]
        print ("Working on folder", folder, "of", len(listdir(directory)))
        for track in tracks: 
            get_track_info(track, dataset, responses)
        #     break
        # break
    print ("Dataset done with len: ", len(dataset), "and responses len: ", len(responses))
    return dataset, responses

def createSVM(trainData, responses):
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
    return svm

hog = cv2.HOGDescriptor(_winSize=(img_shape[1] // cell_size[1] * cell_size[1],
                                  img_shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

# CREATE TRAIN DATA
train_data, train_responses = createDataset(train_directory)
train_data = np.float32([hog.compute(img) for img in train_data])
train_responses = np.array([[a] for a in train_responses])

# CREATE TEST DATA
test_data, test_responses = createDataset(test_directory)
test_data = np.float32([hog.compute(img) for img in test_data])
test_responses = np.array([[a] for a in test_responses])

# CREATE SVM
print ("Creating SVM")
svm = createSVM(train_data, train_responses)
print ("Done creating SVM")

# VERIFY RESULT
result = svm.predict(test_data)[1]
mask = result==test_responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)