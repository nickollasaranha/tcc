# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random
import datetime

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.1
PLATE_HEIGHT_PADDING_FACTOR = 1.1
PLATE_DIMENSION_FACTOR = [3.5, 6]

def drawList(imgLike, lista):

    canvas = np.zeros_like(imgLike)
    contours = [possibleChar.contour for possibleChar in lista]
    cv2.drawContours(canvas, contours, -1, Main.SCALAR_WHITE)

    cv2.imshow("Possible Chars", canvas)
    cv2.waitKey(0)

###################################################################################################
def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    # preprocess to get grayscale and threshold images
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)
    # cv2.imshow("imgThreshScene", imgGrayscaleScene)
    # cv2.waitKey(0)

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    #benchTime = datetime.datetime.now()
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)
    #drawList(imgThreshScene, listOfPossibleCharsInScene)
    # This will print all possible chars found

    #print ("findPossibleCharsInScene took", datetime.datetime.now()-benchTime, "with", len(listOfPossibleCharsInScene), "possible chars.\n")

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    #benchTime = datetime.datetime.now()
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    #print ("findListOfListsOfMatchingChars took", datetime.datetime.now()-benchTime, " with ", len(listOfListsOfMatchingCharsInScene), " possible lists.\n")

    # Attempt to attach plate for each group of matching chars.
    for listOfMatchingChars in (listOfListsOfMatchingCharsInScene):
        #drawList(imgThreshScene, listOfMatchingChars)
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)

        if possiblePlate.imgPlate is not None:
            if PLATE_DIMENSION_FACTOR[0] <= possiblePlate.proportion <= PLATE_DIMENSION_FACTOR[1]:
                listOfPossiblePlates.append(possiblePlate)

    return listOfPossiblePlates

def findPossibleCharsInScene(imgThresh):

    imgThreshCopy = imgThresh.copy()
    _, contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [PossibleChar.PossibleChar(contour) for contour in contours if DetectChars.checkIfPossibleChar(PossibleChar.PossibleChar(contour))]

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # this will be the return value

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY
    
    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights += matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.proportion = imgCropped.shape[1]/imgCropped.shape[0]
    return possiblePlate