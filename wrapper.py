"""Wrapper class for warping 2 images together"""
import os
import cv2
from feat_extractor import FeatureExtractor

class Wrapper:
    """Wrapper class responsible for wrapping images"""

    def __init__(self, output_dir=None):
        """Initalize the wrapper class with a feature extractor."""
        self.extractor = FeatureExtractor()
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.joins(os.getcwd(), "output")
    
    def match_descriptors(self, des1, des2, ratio=0.75):
        """Get matching descriptors between 2 images."""
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # use lowe's test
        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < n.distance * ratio:
                good_matches.append(m)
        return good_matches

    def find_homography(self, kp1, kp2, matches, max_error=4.0):
        """"Finds the homography to fit the 2 images"""
        pts_a = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, status = cv2.find_homography(pts_a, pts_b,
                                        cv.RANSAC, max_error)
        return H, status.flatten()

    def match(self, img_a, img_b):
        kp1, des1, _ = self.FeatureExtractor.detect_and_describe(img_a)
        kp2, des2, _ = self.FeatureExtractor.detect_and_describe(img_b)
        matches = self.match_descriptors(des1=des1, des2=des2)
        if len(matches) < 10:
            print("Not enough matches:", len(matches))
            return None
        H, status = self.find_homography(kp1=kp1, kp2=kp2, matches=matches)
        hB, wB = img_b.shape[:2]
        warpedA = cv2.warpPerspective(img_a, H, (wB, hB))
        maskA = (warpedA > 0).all(axis=2)
        maskB = (img_b    > 0).all(axis=2)
        blend = warpedA.copy()
        # average in overlap
        overlap = maskA & maskB
        blend[overlap] = ((warpedA[overlap].astype(np.float32)
                           + img_b[overlap].astype(np.float32)) / 2).astype(np.uint8)
        blend[maskB & ~maskA] = img_b[maskB & ~maskA]
        return blend
