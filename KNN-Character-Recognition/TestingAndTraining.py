import cv2
import operator
import numpy as np
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class contourData():

    npaContour = None
    boundingRect = None
    rectX = 0
    rectY = 0
    rectWidth = 0
    rectHeight = 0
    area = 0.0

    def calcBoundingRect(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.rectX = intX
        self.rectY = intY
        self.rectWidth = intWidth
        self.rectHeight = intHeight

    def contourValid(self):
        if self.area < MIN_CONTOUR_AREA:
            return False
        return True

def main():
    ContoursData = []
    validContours = []

    try:
        npaClassifications = np.loadtxt("Classifications.txt", np.float32)
    except:
        print("Error! Unable to open the File --> Classification.txt, Exiting Program\n")
        os.system("pause")
        return
    #end

    try:
        npaFlattenedImages = np.loadtxt("FlattenedImages.txt", np.float32)
    except:
        print("Error! Unable to open the File --> flattened_images.txt, Exiting Program\n")
        os.system("pause")
        return
    #end


    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest = cv2.ml.KNearest_create() 

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imageTestingNumbers = cv2.imread("test1.png")

    if imageTestingNumbers is None:
        print("Error! Image file could not be read \n\n")
        os.system("pause")
        return

    #end

    imageGray = cv2.cvtColor(imageTestingNumbers, cv2.COLOR_BGR2GRAY)
    imageBlurred = cv2.GaussianBlur(imageGray, (5,5), 0)

    imageThresh = cv2.adaptiveThreshold(imageBlurred,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        11,
                                        2)
    

    imageThreshCpy = imageThresh.copy()

    imgContours, npaContours, npaHierarchy = cv2.findContours(imageThreshCpy,
                                                              cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:
        contourWithData = contourData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calcBoundingRect()
        contourWithData.area = cv2.contourArea(contourWithData.npaContour)
        ContoursData.append(contourWithData)
    #end

    for contourWithData in ContoursData:
        if contourWithData.contourValid():
            validContours.append(contourWithData)
        #end
    #end

    validContours.sort(key = operator.attrgetter("rectX"))

    finalString = ""

    for contourWithData in validContours:

        cv2.rectangle(imageTestingNumbers,
                      (contourWithData.rectX, contourWithData.rectY),
                      (contourWithData.rectX + contourWithData.rectWidth, contourWithData.rectY + contourWithData.rectHeight),
                      (0, 255, 0),
                       2)

        imageCrop = imageThresh[contourWithData.rectY: contourWithData.rectY + contourWithData.rectHeight,
                            contourWithData.rectX: contourWithData.rectX + contourWithData.rectWidth]

        imageCropResized = cv2.resize(imageCrop,(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        npaCropResized = imageCropResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        npaCropResized = np.float32(npaCropResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaCropResized, k = 1)

        currentChar = str(chr(int(npaResults[0][0])))

        finalString = finalString + currentChar

    #end

    print("\n" +finalString+ "\n")

    cv2.imshow("imgTestingNumbers", imageTestingNumbers)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()

    

    
                       

          
                                     

    
            
        
        
    


        
    
