# DetectChars.py

import cv2
import numpy as np
import math
import random
import datetime
import Main
import Preprocess
import PossibleChar
from multiprocessing import Pool
from functools import partial

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

MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 0.8

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 2
MAX_CHANGE_IN_WIDTH = 2.5
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 5

RESIZED_CHAR_IMAGE_WIDTH = 16
RESIZED_CHAR_IMAGE_HEIGHT = 24

MIN_CONTOUR_AREA = 50

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
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
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        # if no groups of matching chars were found in the plate
        if (len(listOfListsOfMatchingCharsInPlate) == 0):

            possiblePlate.strChars = ""
            continue						# go back to top of for loop
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
# end function

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [PossibleChar.PossibleChar(cnt) for cnt in contours if checkIfPossibleChar(PossibleChar.PossibleChar(cnt))]
# end function

###################################################################################################
# this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
# note that we are not (yet) comparing the char to other chars to look for a group
def checkIfPossibleChar(possibleChar):

    # print (possibleChar.intBoundingRectWidth, possibleChar.intBoundingRectHeight, possibleChar.intBoundingRectArea, possibleChar.fltAspectRatio)
    # imgContours = np.zeros((1080, 1920, 3), np.uint8)
    # cv2.drawContours(imgContours, possibleChar.contour, -1, Main.SCALAR_WHITE)
    # cv2.imshow("imgCountours", imgContours)
    # cv2.waitKey(0)

    if (possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and 
        possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        possibleChar.intBoundingRectArea > MIN_CONTOUR_AREA and
        (MIN_ASPECT_RATIO <= possibleChar.fltAspectRatio <= MAX_ASPECT_RATIO)
        ):
        #print ("ok")
        return True
    else:
        #print ("no")
        return False

###################################################################################################
# with this function, we start off with all the possible chars in one big list
# the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
# note that chars that are not found to be in a group of matches do not need to be considered further
def findListOfListsOfMatchingChars(listOfPossibleChars):

    # this will be the return value
    listOfListsOfMatchingChars = []

    # for each possible char in the one big list of chars
    for possibleChar in listOfPossibleChars:

        # find all chars in the big list that match the current char
        listOfMatchingChars = [possibleChar] + findListOfMatchingChars(possibleChar, listOfPossibleChars)

        # if current possible list of matching chars is not long enough to constitute a possible plate
        # jump back to the top of the for loop and try again with next char, note that it's not necessary
        # to save the list in any way since it did not have enough chars to be a possible plate
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS: continue

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        # so add to our list of lists of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        # recursive call
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      
        # for each list of matching chars found by recursive call
        # add to our original list of lists of matching chars
        # end for
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

        break

    return listOfListsOfMatchingChars
# end function

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

def findListOfMatchingChars(possibleChar, listOfChars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    return [possibleMatchingChar for possibleMatchingChar in listOfChars if matchChars(possibleChar, possibleMatchingChar)]

###################################################################################################
# Euclidian distance
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
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
# end function

###################################################################################################
# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # if current char and other char are not the same char . . .
                                                                            # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # if we get in here we have found overlapping chars
                                # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # then remove current char
                        # end if
                    else:                                                                       # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################
# this is where we apply the actual char recognition
def recognizeCharsInPlate(possiblePlate, listOfMatchingChars):
    strChars = ""               # this will be the return value, the chars in the lic plate

    imgThresh = possiblePlate.imgGrayscale

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # make color version of threshold image so we can draw contours in color on it

    for currentChar in listOfMatchingChars:                                         # for each char in plate

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
        strChars = strChars + predict_char
    # end for

    return strChars
# end function