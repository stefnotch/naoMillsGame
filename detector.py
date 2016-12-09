import cv2
import numpy as np

from random import randint

#Detecting duplicates
MAX_DUPLICATE_DELTA = 1.1
#Minimum area
MIN_AREA = 100

#Checking if 2 areas are equal
MAX_AREA_DELTA = 1.3
NONEXISTENT = -404


#I like GLSL, so I decided to implement this function...
def clamp(num, minValue, maxValue):
    return min(max(num, minValue), maxValue)

#The maximum ratio delta should be something like 1.2
#Only works for positive numbers
def approxEqual(num1, num2, maxRatioDelta):
    ratio = num1 / num2
    if(num2 > num1):
        ratio = 1 / ratio
    return ratio < maxRatioDelta

"""
video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
returnValue, colorImg = video.read()
"""
colorImg = cv2.imread('2.jpg')
colorImg = cv2.resize(colorImg, (0,0), fx=0.5, fy=0.5)



img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
img = cv2.blur(img, (3, 3))

edges = cv2.Canny(img, 70, 200)

#edges = cv2.bitwise_not(edges)
#ret2,edges = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#edges = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 30)

#cv2.imshow("Can I?", edges)

magicWhatDoesThisDo, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''
>>> hierarchy
array([[[ 7, -1,  1, -1],
        [-1, -1,  2,  0],
        [-1, -1,  3,  1],
        [-1, -1,  4,  2],
        [-1, -1,  5,  3],
        [ 6, -1, -1,  4],
        [-1,  5, -1,  4],
        [ 8,  0, -1, -1],
        [-1,  7, -1, -1]]])

[ 7, -1,  1, -1]
vvvvvvvvvvvvvvv
Contour 0: [NextOtherContourSameLevel, PrevOtherContourSameLevel, ChildContour, ParentContour]
'''

#h, w = img.shape
#contourImg = np.zeros((h, w))
#colorImg = cv2.drawContours(colorImg, contours[0], -1, (0, 0, 255), 1)

colorImg[:] = (0, 0, 0)

hierarchy = hierarchy[0]

def removeContour(hierarchy, contourIndex):
    global NONEXISTENT
    parentContour = hierarchy[contourIndex][3]
    childContour = hierarchy[contourIndex][2]
    prevContour = hierarchy[contourIndex][1]
    nextContour = hierarchy[contourIndex][0]
    #Remove all references to this contour
    hierarchy[parentContour][2] = childContour
    hierarchy[childContour][3] = parentContour
    hierarchy[prevContour][2] = nextContour
    hierarchy[nextContour][3] = prevContour
    hierarchy[i][0] = NONEXISTENT
    hierarchy[i][1] = NONEXISTENT
    hierarchy[i][2] = NONEXISTENT
    hierarchy[i][3] = NONEXISTENT

    
#Get rid of the crappy duplicates and the too small contours
for i in range(len(contours)):#{
    parentContour = hierarchy[i][3]
    childContour = hierarchy[i][2]
    area = cv2.contourArea(contours[i])
    #Child has no adjacent contours
    #And the child has an area similar to the area of it's parent
    if(hierarchy[i][0] == -1 and approxEqual(cv2.contourArea(contours[parentContour]), area, MAX_DUPLICATE_DELTA)):#{
        #print("removed: " + str(i))
        removeContour(hierarchy, i)
    #}
    if(area < MIN_AREA):#{
        removeContour(hierarchy, i)
    #}
#}





#Note to self: Implement a iterative algorithm for looping over the tree, which acts like a recursive one
#Down: Always left branch until you hit the bottom
#Up: Move up until you can go one branch to the right
#Repeat!
"""
DOWN = 1
UP = -1
END = 0
dir = 1
index = 0
while(dir != END):
    if(dir == DOWN):
        while(hierarchy[index][2] != -1):
            index = hierarchy[index][2]
        #Now we have reached the bottom
        dir = UP
    elif(dir == UP):
        #As long as there is no right sibling
        while(hierarchy[index][0] == -1):
            #If we reached the top, stop this!
            if(hierarchy[index][3] == -1):
                dir = END
                break
            else
                #go up
                index = hierarchy[index][3]
        if(dir != END):
            #Now we reached the level above (children at the left)
            dir = DOWN
        
"""
#Stores the parents of all boards
possibleBoards = []

for i in range(len(contours)):#{
    #If the contour exists and is the leftmost contour
    if(hierarchy[i][0] != NONEXISTENT and hierarchy[i][1] == -1):#{
        #Contour
        cnt = contours[i]
        #Area
        area = cv2.contourArea(cnt)

        #Quick heuristics
        #Number of siblings check
        numOfSiblings = 0
        nextSiblingIndex = i
        #0.08 -> too large (unless you want all the contours to be triangles, except the one in the middle)
        #0.07 -> too large
        #0.06 -> too large
        #0.05 = :D
        #0.04 = :D
        #0.03 = :D
        #0.02 -> too small
        #0.01 -> too small
        
        #Number of corners check
        sixCorners = 0
        fourCorners = 0
        
        while(True):#{
            if(hierarchy[nextSiblingIndex][0] != NONEXISTENT):
                numOfSiblings += 1
                #Calculates the contour perimeter or the curve length
                perimeter = cv2.arcLength(contours[nextSiblingIndex], True)
                epsil = 0.04
                approx = cv2.approxPolyDP(contours[nextSiblingIndex], perimeter * epsil, True) 
                contours[nextSiblingIndex] = approx
                if(len(approx) == 6):
                    sixCorners += 1
                elif(len(approx) == 4):
                    fourCorners += 1
                else:
                    print("Number of corners: " + str(len(approx)) + " Index: " + str(nextSiblingIndex));
                            
            #Next sibling
            nextSiblingIndex = hierarchy[nextSiblingIndex][0]
            if(nextSiblingIndex == -1 or numOfSiblings > 9):
                break
                
        #}
        if(numOfSiblings != 9):
            continue #Pretty much the same as "return"
        if(sixCorners != 8):
            continue
        if(fourCorners != 1):
            continue

        possibleBoards.append(hierarchy[i][3])
        
        print("Number of siblings" + str(numOfSiblings))
        print("Corners: " + str(sixCorners) + "*6 " +  str(fourCorners) + "*4");
        
        #End of quick heuristics
        """

        #COMPARE EACH AREA WITH EACH AREA!!!!
        #4 areas equal to 4 other areas --> one match
        
        print("Starting")
        print(area)
        #How many contours are there with the same area?
        numOfSameAreaContours = 1
        #The index of the next contour (same level)
        nextContourIndex = hierarchy[i][0]
        while(nextContourIndex != -1):
            nextContour = contours[nextContourIndex]
            nextArea = cv2.contourArea(nextContour)
            print(str(nextArea) + "str" +str(approxEqual(area, nextArea, MAX_AREA_DELTA)))
            if(approxEqual(area, nextArea, MAX_AREA_DELTA)):
                numOfSameAreaContours += 1
                print("yo" + str(numOfSameAreaContours));

            nextContourIndex = hierarchy[nextContourIndex][0]
               
            if(numOfSameAreaContours >= 4):
                print("Yay" + str(numOfSameAreaContours))
        """
    #}
#}


for i in range(len(contours)):
    if(hierarchy[i][0] != NONEXISTENT):
        hue = i/len(contours) * 255
        hsvColor = np.uint8([[[hue, 255, 255]]])
        
        rgb = cv2.cvtColor(hsvColor, cv2.COLOR_HSV2BGR)

        #The countour that we will draw
        cnt = contours[i]
        colorImg = cv2.drawContours(colorImg, [cnt], 0, (int(rgb[0][0][0]), int(rgb[0][0][1]), int(rgb[0][0][2])), 1)
        colorImg = cv2.drawContours(colorImg, cnt, -1, (0,0,255), 2)
        #colorImg = cv2.drawContours(colorImg, [cnt], 0, (randint(0,255),randint(0,255),randint(0,255)), cv2.FILLED)


cv2.imshow("Edgelords FTW" , colorImg)


while True: #{
    if cv2.waitKey(50) & 0xFF == 27:
        break
#}


cv2.destroyAllWindows()
