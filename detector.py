import cv2
import numpy as np


"""
TODO:
1) Decrement epsil (for approxContours) if the board isn't found (loose constraints --> serious constraints)
2) Check out the Nao's camera (hopefully it is better than my crap cam)
3) Figure out the best way to detect the board

"""
global tester


from random import randint
WIDTH = 640
HEIGHT = 480
#Detecting duplicates
MAX_DUPLICATE_DELTA = 1.1
#Minimum area
MIN_AREA = 0.005 * WIDTH * HEIGHT

#Checking if 2 areas are equal
MAX_AREA_DELTA = 1.3
NONEXISTENT = -404
#0 = Fujitsu Webcam
#2 = external cam
video = cv2.VideoCapture(0)
if(not video.isOpened()):
    video.open(-1)
    print("Most likely, this won't work")

video.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

def main():
    global MAX_DUPLICATE_DELTA
    global MIN_AREA
    global MAX_AREA_DELTA
    global NONEXISTENT

    global video
    returnValue, colorImg = video.read()
    if(returnValue == False):
        print("No video?")
        return

    #colorImg = cv2.imread('2.jpg')
    #colorImg = cv2.resize(colorImg, (0,0), fx=0.5, fy=0.5)

    #cv2.imshow("Color", colorImg)

    img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

    blurSize = 5
    img = cv2.GaussianBlur(img, (blurSize, blurSize), 0, 0)
    """
    #Testing out the different image blurring algorithms
    h, w = img.shape
    dst = np.zeros((h, w))

    dst = cv2.blur(img, (blurSize, blurSize))
    cv2.imshow("test", dst)
    dst = cv2.GaussianBlur(img, (blurSize, blurSize), 0, 0)
    cv2.imshow("testG", dst)
    dst = cv2.medianBlur(img, blurSize)
    cv2.imshow("testM", dst)
    """


    #edges = cv2.Canny(img, 70, 100)

    """
    #Auto canny
    sigma = 0.33
    # compute the median of the single channel pixel intensities
    v = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    # return the edged image
    """

    #edges = cv2.bitwise_not(edges)
    #ret2,edges = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #edges = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 33, 0)

    """
    #Also works reasonably well without the laplacian filter
    edges = cv2.Laplacian(img,cv2.CV_8U, scale = 5)
    edges = cv2.blur(edges, (3, 3))
    edges = cv2.threshold(img,170,255,cv2.THRESH_BINARY)[1] #!!!![1]!!!!
    cv2.imshow("Lap", edges)
    """
    #TODO Tweak those parameters (the last 2)
    edges = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    edges = cv2.bitwise_not(edges)

    kernel = np.ones((5,5),np.uint8)
    #edges = cv2.dilate(edges,kernel,iterations = 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Adaptive Threshold", edges)

    magicWhatDoesThisDo, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(hierarchy is None):
        print("No edges")
        return

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

    cleanUpHierarchy(hierarchy, contours)
    #showContours(hierarchy, contours, colorImg, 0)

    #indexes & indices are both purr-fectly acceptable words
    pawsibleBoardIndices = findBoards(hierarchy, contours)

    #Perspective transformation
    if(len(pawsibleBoardIndices) >= 1):
        boardImageSize = 300
        boardImage = np.zeros((boardImageSize, boardImageSize))
        boardContour = contours[pawsibleBoardIndices[0]]
        
        fromPoints = np.float32([boardContour[0,0], boardContour[1,0], boardContour[2,0], boardContour[3,0]])
        toPoints = np.float32([[0, 0], [0, boardImageSize], [boardImageSize, boardImageSize], [boardImageSize, 0]])
        matrix = cv2.getPerspectiveTransform(fromPoints, toPoints)

        boardImage = cv2.warpPerspective(img, matrix, (boardImageSize,boardImageSize))
        cv2.imshow("Warp Drive" , boardImage)


    
    showContours(hierarchy, contours, colorImg, 1)

    for boardIndex in pawsibleBoardIndices:
        #Draw the board
        colorImg = cv2.drawContours(colorImg, [contours[boardIndex]], -1, (0,0,255), 20)
        #With all of it's children
        childIndex = hierarchy[boardIndex,2]
        while(childIndex != -1):
            colorImg = cv2.drawContours(colorImg, [contours[childIndex]], 0, (randint(0,255),randint(0,255),randint(0,255)), 1)
            childIndex = hierarchy[childIndex,0]

    """
    for i in range(len(contours)):
        if(hierarchy[i,0] != NONEXISTENT):
            hue = i/len(contours) * 255
            hsvColor = np.uint8([[[hue, 255, 255]]])

            rgb = cv2.cvtColor(hsvColor, cv2.COLOR_HSV2BGR)

            #The countour that we will draw
            cnt = contours[i]
            colorImg = cv2.drawContours(colorImg, [cnt], 0, (int(rgb[0][0][0]), int(rgb[0][0][1]), int(rgb[0][0][2])), 1)
            colorImg = cv2.drawContours(colorImg, cnt, -1, (0,0,255), 2)
            #colorImg = cv2.drawContours(colorImg, [cnt], 0, (randint(0,255),randint(0,255),randint(0,255)), cv2.FILLED)
    """

    cv2.imshow("Edgelords FTW" , colorImg)
#}

def showContours(hierarchy, contours, img, showID = 0):
    for i in range(len(contours)):
        if(hierarchy[i,0] != NONEXISTENT):
            #The countour that we will draw
            cnt = contours[i]
            img = cv2.drawContours(img, [cnt], 0, (randint(0,255),randint(0,255),randint(0,255)), 1)
            img = cv2.drawContours(img, cnt, -1, (0,0,255), 2)
    #cv2.imshow("ShowMe" + str(showID), img)

#I like GLSL, so I decided to implement this function...
def clamp(num, minValue, maxValue):
    return min(max(num, minValue), maxValue)

#The maximum ratio delta should be something like 1.2
#Only works for positive numbers
def approxEqual(num1, num2, maxRatioDelta):
    if(num2 == 0 or num1 == 0): return False
    ratio = num1 / num2
    if(num2 > num1):
        ratio = 1 / ratio
    return ratio < maxRatioDelta

def invalidateContour(hierarchy, contourIndex):
    global NONEXISTENT
    hierarchy[contourIndex,0] = NONEXISTENT
    hierarchy[contourIndex,1] = NONEXISTENT
    hierarchy[contourIndex,2] = NONEXISTENT
    hierarchy[contourIndex,3] = NONEXISTENT

def cleanUpHierarchy(hierarchy, contours):#{
    """
    Gets rid of the crappy duplicates and the too small contours
    """
    global NONEXISTENT
    for i in range(len(contours)):#{
        parentContour = hierarchy[i,3]
        childContour = hierarchy[i,2]
        nextContour = hierarchy[i,0]
        prevContour = hierarchy[i,1]
        area = cv2.contourArea(contours[i])
        #Has no adjacent contours
        #Has an area similar to the area of it's parent
        if(hierarchy[i,0] == -1 and hierarchy[i,1] == -1 and approxEqual(cv2.contourArea(contours[parentContour]), area, MAX_DUPLICATE_DELTA)):#{
            #print("removed: " + str(i))
            if(parentContour != -1):
                hierarchy[parentContour,2] = childContour
            if(childContour != -1):
                hierarchy[childContour,3] = parentContour
            invalidateContour(hierarchy, i)
        #}

        if(area < MIN_AREA):#{
            #Fix the parent
            if(parentContour != -1):
                if(prevContour != -1):
                    hierarchy[parentContour,2] = prevContour
                else:
                    hierarchy[parentContour,2] = nextContour

            #Fix the previous and next contours
            if(prevContour != -1):
                hierarchy[prevContour,0] = nextContour
            if(nextContour != -1):
                hierarchy[nextContour,1] = prevContour
            #We don't need to deal with the children, because they will get removed in the later iterations (Their area is too small as well)
            invalidateContour(hierarchy, i)
        #}
    #}
#}

def findBoards(hierarchy, contours):#{
    """
    Finds all possible boards and returns them
    Needs a cleaned up hierarchy
    Also messes around with the contours
    """
    global NONEXISTENT

    #Stores the parents of all boards
    possibleBoards = []
    for i in range(len(contours)):#{
        #If the contour exists and is the leftmost contour
        if(hierarchy[i,0] != NONEXISTENT and hierarchy[i,1] == -1):#{
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
            epsil = 0.03
            #Number of corners check
            sixCorners = 0
            fourCorners = 0

            while(True):#{
                if(hierarchy[nextSiblingIndex,0] != NONEXISTENT):
                    numOfSiblings += 1
                    #Calculates the contour perimeter or the curve length

                    perimeter = cv2.arcLength(contours[nextSiblingIndex], True)
                    approx = cv2.approxPolyDP(contours[nextSiblingIndex], perimeter * epsil, True)
                    #approx = approxPolyFixCorners(approx, 60)
                    contours[nextSiblingIndex] = approx

                    if(len(approx) == 6):
                        sixCorners += 1
                    elif(len(approx) == 4):
                        fourCorners += 1
                    else:
                        pass
                        #print("Number of corners: " + str(len(approx)) + " Index: " + str(nextSiblingIndex));

                #Next sibling
                nextSiblingIndex = hierarchy[nextSiblingIndex,0]
                if(nextSiblingIndex == -1 or numOfSiblings > 9):
                    break

            #}
            if(numOfSiblings != 9):
                continue #Pretty much the same as "return"
            if(sixCorners != 8):
                continue
            if(fourCorners != 1):
                continue

            #Parent check
            boardIndex = hierarchy[i,3]
            perimeter = cv2.arcLength(contours[boardIndex], True)
            approx = cv2.approxPolyDP(contours[boardIndex], perimeter * epsil, True)
            #approx = approxPolyFixCorners(approx, 60)
            contours[boardIndex] = approx
            if(len(approx) != 4):
                continue
            contourSimilarAngles(approx, None, 1)
            possibleBoards.append(boardIndex)

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

    return possibleBoards
#}

def contourSimilarAngles(contour, compareTo, angleEpsilon, radians=False):
    """
    compareTo: An array of angles in the range range: [0-180[
    angleEpsilon: An angle
    """
    #if(not radians):
    #    compareTo = np.radians(compareTo)
    
    dpEpsilon = np.cos(np.radians(angleEpsilon))

    #[start:stop:step]
    #[0:-1] -> element 0 to last-1
    vectors = contour[0:-1] - contour[1:]
    #Add the last line (connecting the first and last point)
    vectors = np.append(vectors, [contour[-1] - contour[0]], axis=0)

    xDeltas = vectors[0:,0,0]
    yDeltas = vectors[0:,0,1]
    
    angles = np.arctan2(yDeltas, xDeltas)
    angles = np.mod(angles, np.pi)

    #Now we can start comparing the stuff
    
    global tester
    tester = angles
    return True

    
def approxPolyFixCorners(contour, minAngle):
    """
    Fixes the corners of an approximated contour
    """
    #TODO: this needs to be improved
    
    minDP = np.cos(np.radians(minAngle))
    #corners = np.array([[]], dtype=np.int32)
    length = len(contour)
    if(length < 3):
        return contour

    index = 0
    while(index < length and length >= 3):
        length = len(contour)
        
        prevIndex = (index - 1) % length
        index = index % length
        nextIndex = (index + 1) % length
        #Dot Product
        #Even faster: http://stackoverflow.com/a/14675998/3492994
        prevPoint = contour[prevIndex][0]
        currPoint = contour[index][0]
        nextPoint = contour[nextIndex][0]
        v1 = np.subtract(prevPoint, currPoint)
        v2 = np.subtract(nextPoint, currPoint)
        v1_len = np.linalg.norm(v1)
        v2_len = np.linalg.norm(v2)
        
        v1 = v1 / v1_len
        v2 = v2 / v2_len
        dp = np.dot(v1, v2)
        #If the dot product is greater than the minimum
        #dp at 90 deg = 0
        if(abs(dp) > minDP):
            if(v1_len >= v2_len):
                #Remove the next point
                contour = np.delete(contour, (nextIndex)%length, 0)
            else:
                #Remove the previous point
                contour = np.delete(contour, (prevIndex)%length, 0)
                if(index > 0):
                    index -= 1
        else:
            #Go to the next point
            index += 1
        
    return contour




















while(True): #{
    main()
    if cv2.waitKey(100) & 0xFF == 27:
        break
#}

cv2.destroyAllWindows()


