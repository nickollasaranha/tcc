# DetectChars.py

import cv2
import numpy as np
import math
import random
import datetime

import Main
import Preprocess
import PossibleChar

import threading
# module level variables ##########################################################################
img_shape = [24, 16]
cell_size = (4, 4)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

hog = cv2.HOGDescriptor(_winSize=(img_shape[1] // cell_size[1] * cell_size[1],
                                  img_shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

svm = cv2.ml.SVM_load("svm.dat")

# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 2
MAX_CHANGE_IN_WIDTH = 2.5
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 4

RESIZED_CHAR_IMAGE_WIDTH = 16
RESIZED_CHAR_IMAGE_HEIGHT = 24

MIN_CONTOUR_AREA = 50

def detectCharsInPlates(listOfPossiblePlates, char_aspect_ratio_interval):
    intPlateCounter = 0
    imgContours = None
    contours = []

    # if list of possible plates is empty, return
    if len(listOfPossiblePlates) == 0: return listOfPossiblePlates

    # at this point we can be sure the list of possible plates has at least one plate
    for possiblePlate in listOfPossiblePlates:

        # preprocess to get grayscale and threshold images
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)

        # increase size of plate image for easier viewing and char detection
        possiblePlate.imgGrayscale = cv2.resize(possiblePlate.imgGrayscale, (0, 0), fx = 1.6, fy = 1.6)
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh, char_aspect_ratio_interval)

        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = groupMatchingChars(listOfPossibleCharsInPlate)

        # if no groups of matching chars were found in the plate
        if (len(listOfListsOfMatchingCharsInPlate) == 0):

            possiblePlate.strChars = ""
            continue                        # go back to top of for loop
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # within each list of matching chars
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # and remove inner overlapping chars
        # end for

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate, longestListOfMatchingCharsInPlate)

    # end of big for loop that takes up most of the function

    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh, char_aspect_ratio_interval):
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [PossibleChar.PossibleChar(cnt) for cnt in contours if checkIfPossibleChar(PossibleChar.PossibleChar(cnt), char_aspect_ratio_interval)]

# this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
# note that we are not (yet) comparing the char to other chars to look for a group
def checkIfPossibleChar(possibleChar, char_aspect_ratio_interval):

    if (possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and 
        possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        possibleChar.intBoundingRectArea > MIN_CONTOUR_AREA and
        (char_aspect_ratio_interval[0] <= possibleChar.fltAspectRatio <= char_aspect_ratio_interval[1])
        ):
        #print ("ok")
        return True
    else: return False

# with this function, we start off with all the possible chars in one big list
# the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
# note that chars that are not found to be in a group of matches do not need to be considered further
def groupMatchingChars(listOfPossibleChars):

    visited = set()
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:

        if possibleChar in visited: continue

        # find all matching chars
        listOfMatchingChars = [possibleChar] + [char for char in listOfPossibleChars if ((char not in visited) and (matchChars(possibleChar, char)))]

        # add to visited
        visited.update(listOfMatchingChars)

        # verify if valid
        if len(listOfMatchingChars) >= MIN_NUMBER_OF_MATCHING_CHARS: 
            listOfListsOfMatchingChars.append(listOfMatchingChars)

    return listOfListsOfMatchingChars

# Verify if a char is matching with other, excluding itself
def matchChars(possibleChar, possibleMatchingChar):
    if possibleChar == possibleMatchingChar: return False

    fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
    fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
    fltChangeInArea = float(abs(possibleMatchingChar.area - possibleChar.area)) / float(possibleChar.area)
    fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
    fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

    if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
        fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
        fltChangeInArea < MAX_CHANGE_IN_AREA and
        fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
        fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
        return True

    return False

def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

    return fltAngleInDeg

# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):

    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:
                # if current char and other char have center points at almost the same location                                                                
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)
                    elif otherChar in listOfMatchingCharsWithInnerCharRemoved:               
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)

    return listOfMatchingCharsWithInnerCharRemoved

# this is where we apply the actual char recognition
def recognizeCharsInPlate(possiblePlate, listOfMatchingChars):
    strChars = ""

    imgThresh = possiblePlate.imgGrayscale

    # sort chars from left to right
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    for currentChar in listOfMatchingChars:

        # crop char out of threshold image
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        # resize image, this is necessary for char recognition
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
        imgROI_hog = np.float32([hog.compute(imgROIResized)])
        predict_char = chr(svm.predict(imgROI_hog)[1][0][0])
        # print(predict_char)
        # cv2.imshow("imgROIResized", imgROIResized)
        # cv2.waitKey(0)
        # cv2.imshow("predicted", imgROIResized)
        # print(svm.predict(imgROI_hog))
        # cv2.waitKey(0)

        # append character
        strChars += predict_char

    return strChars