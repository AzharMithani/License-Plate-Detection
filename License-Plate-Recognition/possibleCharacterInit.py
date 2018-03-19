import cv2
import numpy as np
import math

class possibleCharacterInit:

    def __init__(self, _contour):

        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)

        [intA, intB, intBreadth, intLength] = self.boundingRect

        self.intBoundingRectA = intA
        self.intBoundingRectB = intB
        self.intBoundingRectBreadth = intBreadth
        self.intBoundingRectLength = intLength

        self.intBoundingRectArea = self.intBoundingRectBreadth * self.intBoundingRectLength
        self.intCenterA = (self.intBoundingRectA + self.intBoundingRectA + self.intBoundingRectBreadth) / 2
        selr.intCenterB = (self.intBoundingRectB + self.intBoundingRectB + self.intBoundingRectLength) / 2

        self.floatDiagonalSize = math.sqrt((self.intBoundingRectBreadth ** 2) + (self.intBoundingRectLength **2))
        self.floatAspectRatio = float(self.intBoundingRectBreadth) / float(self.intBoundingRectHeight)

    #end

#end
