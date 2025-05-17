"""Feature extractor for aerial mapping."""
import os
import cv2

class FeatureExtractor:
    """
    A class for extracting the feature for an image.
    Computes the keypoints (corners) and descriptors
    for each image.
    """

    def __init__(self, nfeats, orb=True):
        """Handle the initilization of the feature extractor."""
        self.nfeats = nfeats
        if orb:
            self.descriptor = cv2.ORB_create(
                nfeatures=self.nfeats,
                scaleFactor=1.2,
                nlevels=25,
                edgeThreshold=12
            )
        else:
            self.descriptor = cv2.SIFT_create(
                nfeatures=self.nfeats,
                contrastThreshold=0.005,
                edgeThreshold=15,
                nOctaveLayers=12,
                sigma=1.2
            )

    def detect_and_describe(self, image_path):
        """Extract the key pts and the descriptor from an image."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
        kp, des = self.descriptor.detectAndCompute(enhanced_image, None)
        pts = cv2.KeyPoint_convert(kp)
        return kp, des, pts
