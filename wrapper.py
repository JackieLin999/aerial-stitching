"""Wrapper class for warping 2 images together"""
import os
import cv2
import math
from pyproj import Proj
import numpy as np
from feat_extractor import FeatureExtractor
from image_processor import ImageProcessor

class Wrapper:
    """Wrapper class responsible for wrapping images"""

    def __init__(
        self, input_dir,
        output_dir=None, focal_length=4800,
        principal_x=2254, principal_y=2048,
        nfeats=5000, sensor_width=0.01127
        ):
        """Initalize the wrapper class with a feature extractor."""
        self.images_dir = (
            input_dir or os.path.join(os.getcwd(), "images", "test")
        )

        self.img_processor = ImageProcessor(
            images_dir=self.images_dir,
            focal_length=focal_length,
            principal_x=principal_x,
            principal_y=principal_y
        )

        # photos holds all file path for the images
        photos = self.img_processor.process_images()

        self.gps_info = {
            "base_gps": None,
            "ground_resolution": None,
            "utm_zone": None
        }

        self._init_gps_infos(sensor_width)

        self.image_positions = {}
        self._est_imgs_pos(imgs=photos)

        self.extractor = FeatureExtractor(nfeats=nfeats)

        self.focal_length = focal_length


        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _init_gps_infos(self, sensor_width):
        """Initalize the gps_info variable for the wrapper class."""
        self.gps_info["base_gps"] = self.img_processor.get_base_gps()
        self.gps_info["ground_resolution"] = self._calculate_ground_res(sensor_width)
        self.gps_info["utm_zone"] = self._calculate_utm_zone(self.gps_info["base_gps"]["long"])
        print("Sucessfully initalize gps infos")

    def _calculate_ground_res(self, sensor_width):
        img_width = self.img_processor.get_img_size()[0]
        return (sensor_width * self.gps_info["base_gps"]["alt"]) / (self.focal_length * img_width) 

    def _calculate_utm_zone(self, longitude):
        """Calculate the utm zone the drone is in."""
        return int(math.floor((longitude + 180) / 6) + 1)

    def _est_imgs_pos(self, imgs):
        """Add UTM coordinate to each image."""
        # an object for conversion between degree to utm x, y
        proj = Proj(proj='utm', zone=self.gps_info["utm_zone"], ellps='WGS84')
        base_lat = self.gps_info["base_gps"]["lat"]
        base_long = self.gps_info["base_gps"]["long"]
        base_x, base_y = proj(base_lat, base_long)
        for img in imgs:
            curr_gps = self.img_processor.get_gps(image_path=img)
            x, y = proj(curr_gps['long'], curr_gps['lat'])
            self.image_positions[img] = {
                'utm_x': x,
                'utm_y': y,
                'gps': curr_gps,
                'offset': (x - base_x, y - base_y)
            }
        print("Sucessfully Complete estimating image positions")

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
                                        cv2.RANSAC, max_error)
        return H, status.flatten()

    def match(self, img_a, img_b):
        kp1, des1, _ = self.extractor.detect_and_describe(img_a)
        kp2, des2, _ = self.extractor.detect_and_describe(img_b)
        matches = self.match_descriptors(des1=des1, des2=des2)
        if len(matches) < 10:
            print("Not enough matches:", len(matches))
            return None
        H, status = self.find_homography(kp1=kp1, kp2=kp2, matches=matches)
        return H
        # hB, wB = img_b.shape[:2]
        # warpedA = cv2.warpPerspective(img_a, H, (wB, hB))
        # maskA = (warpedA > 0).all(axis=2)
        # maskB = (img_b > 0).all(axis=2)
        # blend = warpedA.copy()
        # # average in overlap
        # overlap = maskA & maskB
        # blend[overlap] = ((warpedA[overlap].astype(np.float32)
        #                    + img_b[overlap].astype(np.float32)) / 2).astype(np.uint8)
        # blend[maskB & ~maskA] = img_b[maskB & ~maskA]
        # return blend
