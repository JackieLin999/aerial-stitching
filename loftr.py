"""The LoFTR class"""
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
from kornia.feature import LoFTR


class LoFTR:
    """A class of the LoFTR for extracting kp and des"""
    
    def __init__(self):
        """Init the LoFTR class."""
        self.matcher = KF.LoFTR(pretrained="outdoor")

    def preprocess_img(self, img_path, size=(512, 512)):
        """Load and preprocess an image."""
        img = K.io.load_image(image_path, K.io.ImageLoadType.RGB32)[None, ...]
        img = K.geometry.resize(img, size, antialias=True)
        return img

    def match(self, img_1, img_2):
        """Match features between 2 images"""
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img_1),
            "image1": K.color.rgb_to_grayscale(img2_2),
        }
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)
        return correspondences

    def extract_kps(self, correspondences):
        """Extract the kps from correspondeces"""
        keypoints0 = correspondences["keypoints0"].cpu().numpy()
        keypoints1 = correspondences["keypoints1"].cpu().numpy()
        confidence_scores = correspondences["confidence"].cpu().numpy()
        return keypoints0, keypoints1, confidence_scores
    