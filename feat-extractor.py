"""Feature extractor for aerial mapping."""
import os
import cv2

class FeatureExtractor:
    """
    A class for extracting the feature for an image.
    Computes the keypoints (corners) and descriptors
    for each image.
    """
    
    def __init__(self, nfeats=5000):
        """Handle the initilization of the feature extractor."""
        self.nfeats = nfeats
    
    def detect_and_describe(self, image):
        """Extract the key pts and the descriptor from an image.""""
        descriptor = cv2.ORB_create(nfeatures=self.nfeats)
        kp, des = descriptor.detectAndCompute(image, None)
        pts = cv2.KeyPoint_convert(kp)
        return kp, des, pts
