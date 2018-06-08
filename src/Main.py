# Main.py

import cv2
import numpy as np
import os
import re
from os import listdir

import DetectChars
import DetectPlates
import PossiblePlate
import time

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

MIN_CHARS_PLATE = 5
MAX_CHARS_PLATE = 7
showSteps = False

###################################################################################################
def recognize(imgOriginalScene, char_aspect_ratio_interval, plate_aspect_ratio_interval):

    if imgOriginalScene is None:
        print ("\nerror: Couldn't load image file.\n\n")
        os.system("pause")
        return

    #imgOriginalScene = cv2.resize(imgOriginalScene, (1280, 720), interpolation = cv2.INTER_CUBIC)
    # Start plate recognition
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene, char_aspect_ratio_interval, plate_aspect_ratio_interval)

    # detect chars in plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates, char_aspect_ratio_interval)

    listOfPossiblePlates = [plate for plate in listOfPossiblePlates if MIN_CHARS_PLATE <= len(plate.strChars) <= MAX_CHARS_PLATE]
    if len(listOfPossiblePlates) == 0: return ""

    # if we get in here list of possible plates has at leat one plate
    # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
    listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

    # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order)
    # is the actual plate
    licPlate = listOfPossiblePlates[0]

    licPlate.strChars = heuristic(licPlate.strChars)

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
        "P": "8",
        "Q": "0",
        "D": "0"
    }
    return switcher.get(consoante, str(consoante))

def heuristic(strChar):

    consoantes = list(strChar[0:3])
    numeros = list(strChar[3:])

    consoantes = [to_consoante(i) for i in consoantes]
    numeros = [to_digit(i) for i in numeros]

    return "".join(consoantes+numeros)

def run(char_aspect_ratio_interval, plate_aspect_ratio_interval):

    directory = "C:\\Users\\Nickollas Aranha\\Documents\\tcc\\database\\training\\"
    text_extension = ".txt"
    image_extension = ".png"

    # Benchmark and results vars
    total_tracks = 0
    positive_tracks = 0
    total_images = 0
    positive_images = 0

    time_all = time.time()
    time_worst = 0
    time_best = time.time()

    # For single time running purposes
    print(recognize(cv2.imread("12.png"), char_aspect_ratio_interval, plate_aspect_ratio_interval), time.time()-time_all)
    return

    for folder in listdir(directory):

        tracks = [os.path.splitext(directory+folder+"\\"+track)[0] for track in listdir(directory+folder) if os.path.splitext(directory+folder+"\\"+track)[1]==".txt"]
        #print ("Working on folder", folder, "of", len(listdir(directory)))
        flag = True
        total_tracks+=1
        for track in tracks:
            total_images+=1
            file = open(track+text_extension)
            time_now = time.time()
            predictedPlate = recognize(cv2.imread(track+image_extension), char_aspect_ratio_interval, plate_aspect_ratio_interval)
            time_end = time.time()

            if (time_end-time_now) < time_best: time_best = (time_end-time_now)
            if (time_end-time_now) > time_best: time_worst = (time_end-time_now)

            # Get all chars position
            lines = file.readlines()

            # Get plate number
            plate_number = re.sub("[-\nplate: ]", "", lines[6])
            if plate_number == predictedPlate:
                positive_images+=1
                if flag:
                    positive_tracks+=1
                    flag = False

            #print ("Track", track, "predictedPlate:", predictedPlate, "correct:", "".join(plate_number))

    result_image = 0 if total_images == 0 else float(positive_images)/float(total_images)
    result_tracks = 0 if total_tracks == 0 else float(positive_tracks)/float(total_tracks)

    # print header and info below
    print ("Min Plate Aspect Ratio;Max Plate Aspect Ratio;Min Char Aspect Ratio;Max Char Aspect Ratio;Time running;Best Time;Worst Time;Total pictures;Total tracks;Correct Images;Correct tracks")
    print (plate_aspect_ratio_interval[0], ";", plate_aspect_ratio_interval[1], ";", char_aspect_ratio_interval[0], ";", char_aspect_ratio_interval[1], ";", time.time()-time_all, ";", time_best, ";", time_worst, ";", total_images, ";", total_tracks, ";", result_image, ";", result_tracks)

if __name__ == "__main__":
    
    run([0.2, 0.8], [3.5, 6])
    # # this first text verify the aspect ratio of possible chars.
    # CHAR_ASPECT_RATIO_INTERVAL = [float(x)/float(10) for x in range(11)]
    # PLATE_ASPECT_RATIO_INTERVAL = [float(x)/float(10) for x in range(20, 101, 5)]

    # # Char Aspect Ratio
    # for min_char_aspect_ratio in CHAR_ASPECT_RATIO_INTERVAL:
    #     for max_char_aspect_ratio in CHAR_ASPECT_RATIO_INTERVAL:
    #         if min_char_aspect_ratio>=max_char_aspect_ratio: continue

    #         # Plate Aspect Ratio
    #         for min_plate_aspect_ratio in PLATE_ASPECT_RATIO_INTERVAL:
    #             for max_plate_aspect_ratio in PLATE_ASPECT_RATIO_INTERVAL:
    #                 if min_plate_aspect_ratio>=max_plate_aspect_ratio: continue
    #                 run([min_char_aspect_ratio, max_char_aspect_ratio], [min_plate_aspect_ratio, max_plate_aspect_ratio])