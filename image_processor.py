import cv2
import os
import numpy as np


class ImageProcessor:
    def __init__(self, images_dir=os.path.join(os.getcwd(), "images/test"),
                focal_length=4800, principal_x=2254, principal_y=2048):
        self.focal_length = focal_length
        self.principal_x = principal_x
        self.principal_y = principal_y
        self.mtx = np.array([
            [self.focal_length, 0, self.principal_x],
            [0, self.focal_length, self.principal_y],
            [0, 0, 1]
        ])
        self.distortion_coeff = np.zeros(5)

        self.images_dir = images_dir
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.photos = [photo for photo in os.listdir(images_dir) if photo.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not self.photos:
            raise ValueError(f"No images found in {self.images_dir}")
        self.img_size = None
        self.new_mtx = None
        self.roi = None
        self.setup_camera()

    def setup_camera(self):
        first_img_path = os.path.join(self.images_dir, self.photos[0])
        img = cv2.imread(first_img_path)
        h, w = img.shape[:2]
        self.img_size = (w, h)
        self.new_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.distortion_coeff, self.img_size, 1, self.img_size)

    def process_images(self):
        x, y, w, h = self.roi
        for i, photo in enumerate(self.photos):
            img_path = os.path.join(self.images_dir, photo)
            img = cv2.imread(img_path)
            processed_photo = cv2.undistort(img, self.mtx, self.distortion_coeff, None, self.new_mtx)
            #crop via regin of interest
            processed_photo = processed_photo[y:y+h, x:x+w]
            processed_image_path = os.path.join(self.temp_dir, f"processed_image_{i}.jpg")
            cv2.imwrite(processed_image_path, processed_photo)
