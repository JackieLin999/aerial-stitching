"""Feature extractor for aerial mapping."""
import os
import cv2

class FeatureExtractor:
    """
    A class for extracting the feature for an image.
    Computes the keypoints (corners) and descriptors
    for each image.
    """

    def __init__(self, nfeats):
        """Handle the initilization of the feature extractor."""
        self.nfeats = nfeats

    def detect_and_describe(self, image_path, orb=True):
        """Extract the key pts and the descriptor from an image."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if orb:
            descriptor = cv2.ORB_create(
                nfeatures=self.nfeats,
                scaleFactor=1.2,
                nlevels=25,
                edgeThreshold=12
            )
        else:
            descriptor = cv2.SIFT_create(
                nfeatures=self.nfeats,
                contrastThreshold=0.01,
                edgeThreshold=20,
                nOctaveLayers=6,
                sigma=1.0
            )

        kp, des = descriptor.detectAndCompute(image, None)
        pts = cv2.KeyPoint_convert(kp)
        return kp, des, pts
