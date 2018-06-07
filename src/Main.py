# Main.py

import cv2
import numpy as np
import os
import re
from os import listdir

import DetectChars
import DetectPlates
import PossiblePlate
import datetime

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def recognize(imgOriginalScene):

    if imgOriginalScene is None:
        print ("\nerror: Couldn't load image file.\n\n")
        os.system("pause")
        return

    #imgOriginalScene = cv2.resize(imgOriginalScene, (1280, 720), interpolation = cv2.INTER_CUBIC)
    # Start plate recognition
    # Start benchmark
    benchTime = datetime.datetime.now()
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    #print ("Detect plates took ", datetime.datetime.now()-benchTime, " with ", len(listOfPossiblePlates), " possible plates.\n")

    benchTime = datetime.datetime.now()
    # detect chars in plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)
    #print ("Detect chars took ", datetime.datetime.now()-benchTime)

    #cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        # Inform the user we couldn't find plates
        #print ("\nCouldn't find license plates.\n")
        return ""
    else:
        # if we get in here list of possible plates has at leat one plate
        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) 
        #is the actual plate
        licPlate = listOfPossiblePlates[0]

        # show crop of plate and threshold of plate
        # cv2.imshow("imgPlate", licPlate.imgPlate)
        # cv2.imshow("imgThresh", licPlate.imgThresh)

        # Error is no license plates found
        if len(licPlate.strChars) == 0:
            #print ("\nno characters were detected\n\n")
            return ""

        # draw red rectangle around plate
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        # Get heuristic
        licPlate.strChars = heuristic(licPlate.strChars)

        # write license plate text to std out
        #print (licPlate.strChars)

        # write license plate text on the image
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           

        # re-show scene and write image file
        #cv2.imshow("imgOriginalScene", imgOriginalScene)
        # cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    # hold windows open until user presses a key
    #cv2.waitKey(0)
    return licPlate.strChars

def to_consoante(digit):
    switcher = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "B",
        "4": "R",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "B"
    }
    return switcher.get(digit, str(digit))

def to_digit(consoante):
    switcher = {
        "O": "0",
        "I": "1",
        "Z": "2",
        "B": "8",
        "S": "5",
        "T": "7",
        "J": "1",
        "U": "0",
        "P": "8"
    }
    return switcher.get(consoante, str(consoante))

def heuristic(strChar):

    consoantes = list(strChar[0:3])
    numeros = list(strChar[3:])

    consoantes = [to_consoante(i) for i in consoantes]
    numeros = [to_digit(i) for i in numeros]

    return "".join(consoantes+numeros)

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    # unpack text size width and height
    textSizeWidth, textSizeHeight = textSize               

    # calculate the lower left origin of the text area
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))

    # based on the text area center, width, and height
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_GREEN, intFontThickness)

###################################################################################################
if __name__ == "__main__":
    
    # directory = "C:\\Users\\Nickollas Aranha\\Documents\\tcc\\database\\testing\\"
    # text_extension = ".txt"
    # image_extension = ".png"
    # count_rights = 0
    # total_counts = 0

    # for folder in listdir(directory):

    #     tracks = [os.path.splitext(directory+folder+"\\"+track)[0] for track in listdir(directory+folder) if os.path.splitext(directory+folder+"\\"+track)[1]==".txt"]
    #     print ("Working on folder", folder, "of", len(listdir(directory)))
    #     flag = True
    #     total_counts+=1
    #     for track in tracks:

    #         file = open(track+text_extension)
    #         predictedPlate = recognize(cv2.imread(track+image_extension))
    #         #imgOriginalScene = cv2.imread()
    #         #imgThreshScene, _ = Preprocess.preprocess(imgOriginalScene)

    #         # # Get all chars position
    #         lines = file.readlines()
    #         # chars_position = [re.sub("[a-z:\n]", "", line).split(" ")[2:] for line in lines[8:]]
    #         # list_char = [get_char(char.copy(), imgThreshScene, dataset) for char in chars_position]

    #         # Get plate number
    #         plate_number = re.sub("[-\nplate: ]", "", lines[6])
    #         if plate_number == predictedPlate and flag:
    #                 count_rights+=1
    #                 flag = False

    #         print ("Track", track, "predictedPlate:", predictedPlate, "correct:", "".join(plate_number)   )

    # print ("Total of:", total_counts, "predicted correted:", count_rights)
    print(recognize(cv2.imread("11.png")))
    #main()