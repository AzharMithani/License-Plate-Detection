import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    imgTrainNos = cv2.imread("training_chars.png")

    if imgTrainNos is None:
        print("error: image not read from the given file! \n\n")
        os.system("pause")
        return
    #end


    imageGray = cv2.cvtColor(imgTrainNos, cv2.COLOR_BGR2GRAY)
    imageBlurred = cv2.GaussianBlur(imageGray, (5,5), 0)

    imageThresh = cv2.adaptiveThreshold(imageBlurred,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        11,
                                        2)

    cv2.imshow("imageThresh", imageThresh)

    imageThreshCopy = imageThresh.copy()

    imageContours, npaContours, npaHierarchy = cv2.findContours(imageThreshCopy,
                                                                cv2.RETR_EXTERNAL,
                                                                cv2.CHAIN_APPROX_SIMPLE)

    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour)> MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(imgTrainNos,
                         (intX, intY),
                         (intX+intW,intY+intH),
                         (0,0,255),
                          2)


            imageCrop = imageThresh[intY:intY+intH, intX:intX+intW]
            imageCropResized = cv2.resize(imageCrop, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))


            cv2.imshow("imageCrop", imageCrop)
            cv2.imshow("imageCropResized", imageCropResized)
            cv2.imshow("training_numbers.png",imgTrainNos)


            intChar = cv2.waitKey(0)

            if intChar == 27:
               sys.exit()
            elif intChar in intValidChars:
                intClassifications.append(intChar)

                npaFlattenedImage = imageCropResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

            #endif
        #endif
    #endfor


    floatClassifications = np.array(intClassifications, np.float32)
    npaClassifications= floatClassifications.reshape(floatClassifications.size, 1)

    print("\n TRAINING DONE!\n")

    np.savetxt("Classifications.txt", npaClassifications)
    np.savetxt("FlattenedImages.txt", npaFlattenedImages)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
# end if
     
            
        
    
