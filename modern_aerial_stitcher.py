"""Stitches images via ml methods."""
import os
import cv2
import numpy as np
from loftr import LoFTR
from usac import USAC

class ModernStitcher:
    """Class for stitching aerial images."""

    def __init__(
        self,
        usac_info,
        camera_info,
        img_size,
        imgs_path,
        output_path="default_out"
    ):
        """Init the modern aerial stitcher"""
        self.loftr = LoFTR(image_size=img_size)
        self.usac = USAC(
            method=usac_info['info'],
            threshold=usac_info['threshold'],
            confidence=usac_info['confidence'],
            max_iterations=usac_info['max_iterations'],
            sigma=usac_info['sigma']
        )
        self.imgs_path = imgs_path
        self.image_files = sorted([
            f for f in os.listdir(imgs_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_and_process(self, img_path, undistort_flag=False):
        """Load and process a single image from bgr to rgb via disk."""
        """OpenCv uses BGR channels and LoFTOR uses RGB channels."""
        path = os.path.join(self.imgs_path, filename)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if undistort_flag:
            k = self.camera_info['intrinsic_matrix']
            dist = self.camera_info['distortion']
            img_bgr = cv2.undistort(img_bgr, k, dist)
        img_resized = cv2.resize(img_bgr, self.img_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return img_rgb

    def match_features(self, image_1, image_2):
        """Compute the key pts."""
        correspondences = self.loftr.match(image_1, image_2)
        kps1, kps2, confidence = self.loftr.extract_kps(correspondences)

    def stitch_images(self, image_path_1, image_path_2, undistort_flag=False):
        """Stitch 2 different images together."""
        image_1 = self.load_and_process(
            image_path=image_path_1,
            undistort_flag=undistort_flag
        )
        image_2 = self.load_and_process(
            image_path=image_path_2,
            undistort_flag=undistort_flag
        )
        kps1, kps2, confidence = self.match_features(image_1, image_2)
        H, mask, metrics = self.usac.eval_and_est_homography(kps1, kps2)

        height, width = image_1.shape
        img2_wrapped = cv2.warpPerspective(image_2, H, (height, width))
        final = np.maximum(image_1, img2_wrapped)
        return final, metrics

    def stitch_all(self):
        """Stitch all of the images"""
        final_image = None
        for i in range(len(self.imgs_files) - 1):
            img1_path = self.imgs_files[i]
            img2_path = self.imgs_files[i + 1]

            if final_image:
                final_image, metrics = self.stitch_images(final_image, img2_path)
            else:
                final_image, metrics = self.stitch_images(img1_path, img2_path)
        output = os.path.join(self.output_dir, "out.tiff")
        cv2.imwrite(output_, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
