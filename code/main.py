import cv2
import numpy as np
import math

# this returns in degrees
def horizontal_angle(line):
    return 0 if line[2]-line[0] == 0 else math.degrees(math.atan(((line[3]-line[1])/(line[2]-line[0]))))

# pixels
def euclid_dist(a, b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def line_width(line):
    return math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2)

def filter_lines_angle(lines, angle_interval):
    return np.array([line for line in lines if angle_interval[0] <= horizontal_angle(line[0]) <= angle_interval[1]])

# This will order from left to right
def order_lines(lines):
    new_lines = []

    for line in lines:
        line = line[0]
        arranged_line = (np.array([line[2], line[3], line[0], line[1]]) if line[0] >= line[2] else line)
        new_lines.append([arranged_line])

    return np.array(new_lines)

def is_continuation(line1, line2, limiar):
    
    end_line1 = [line1[2], line1[3]]
    init_line2 = [line2[0], line2[1]]
    #print (euclid_dist(end_line1, init_line2))
    return True if euclid_dist(end_line1, init_line2) <= limiar else False 

def connect_lines(lines):

    limiar_dist = 10
    limiar_angle = 5

    for index1 in range(len(lines)):
        linha1 = lines[index1][0]
        
        for index2 in range(index1+1, len(lines)):
            linha2 = lines[index2][0]
            if (is_continuation(linha1, linha2, limiar_dist) and abs(horizontal_angle(linha1)-horizontal_angle(linha2))<=limiar_angle):
                #print ("got line ", linha1, "with angle ", horizontal_angle(linha1), " and line ", 
                #    linha2, "with angle ", horizontal_angle(linha2), "and dist is ", 
                #    euclid_dist([linha1[2], linha1[3]], [linha2[0], linha2[1]]), " and diff angle is ",
                #    abs(horizontal_angle(linha1)-horizontal_angle(linha2)))
                new_line = np.array([[linha1[0], linha1[1], linha2[2], linha2[3]]])

                # Verify if new line is ok.
                if abs(horizontal_angle(new_line[0])-horizontal_angle(linha1)) > limiar_angle: continue
                #draw(np.array([linha1, linha2]))

                #print ("old array is \n", lines, "\n new array is \n")
                array = np.delete(lines, [index1, index2], 0)
                
                array = np.insert(array, 0, new_line, 0)
                #print (array)
                return connect_lines(array)

    return lines

def draw(lines):
    drawn_img = lsd.drawSegments(img, lines) 
    #resized = cv2.resize(drawn_img, (0,0), fx=10, fy=10)
    cv2.imshow('image', drawn_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

img = cv2.imread('track1.png', 0)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 1)
roi = img[520:580, 900:1000]

# Detect lines in the image
lines = lsd.detect(img)
horizontal_lines = filter_lines_angle(lines[0], [-10.0, 10.0])
horizontal_lines = order_lines(horizontal_lines)
#horizontal_lines = connect_lines(horizontal_lines)

draw(horizontal_lines)
# for line in horizontal_lines:

#    print (line)
#    #drawn_img = lsd.drawSegments(roi, np.array(line)) 
#    draw(np.array(line))

#TODO:
# Try to remove horizontal segments between bla bla
# Try to remove vertical segments higher than bla bla
# Create a func to check for vertical distances between horizontal lines.
# Create a func to check for horizontal distances between vertical lines.
# Wrap it all and done.