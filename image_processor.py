# pylint: disable=no-member,unpacking-non-sequence
"""Preprocessor class."""
import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS


class ImageProcessor:
    """A class for preprocessing the photos."""

    def __init__(
        self,
        images_dir,
        focal_length,
        principal_x,
        principal_y
    ):
        """Handle the initialization of the image processor class."""
        self.camera = {
            "mtx": np.array([
                [focal_length, 0,            principal_x],
                [0,            focal_length, principal_y],
                [0,            0,            1]
            ], dtype=np.float32),
            "distortion": np.zeros(5, dtype=np.float32),
            # will fill in by setup_camera():
            "img_size": None,
            "new_mtx":  None,
            "roi":      None,
        }

        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.photos = [
            photo
            for photo in os.listdir(images_dir)
            if photo.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        if not self.photos:
            raise ValueError(f"No images found in {self.images_dir}")
        self._setup_camera()

    def _setup_camera(self):
        """Set up the camera intrinsics for cropping."""
        first_img_path = os.path.join(self.images_dir, self.photos[0])
        img = cv2.imread(first_img_path)
        h, w = img.shape[:2]
        self.camera["img_size"] = (w, h)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera['mtx'], self.camera["distortion"],
            self.camera["img_size"], 1, self.camera["img_size"]
        )
        self.camera['new_mtx'] = new_mtx
        self.camera['roi'] = roi

    def process_images(self):
        """Process the aerial photographs."""
        x, y, w, h = self.camera['roi']
        photos_w_geo = {}
        for i, photo in enumerate(self.photos):
            img_path = os.path.join(self.images_dir, photo)
            img = cv2.imread(img_path)
            processed_photo = cv2.undistort(
                img, self.camera['mtx'],
                self.camera["distortion"], None, self.camera["new_mtx"]
            )
            # crop regin of interest
            processed_photo = processed_photo[y:y+h, x:x+w]
            processed_image_path = os.path.join(
                self.temp_dir,
                f"processed_image_{i}.jpg"
            )
            cv2.imwrite(processed_image_path, processed_photo)
            gps = self.get_gps(img_path)
        print("Image Processing successful")
        return self.photos

    def get_gps(self, image_path):
        """Extract the gps coordinates from the image."""
        image = Image.open(image_path)
        # pylint: disable=W0212
        exif_data = image._getexif()
        gps_info_tag = 0x8825
        gps_data = exif_data.get(gps_info_tag)
        gps = {}
        for gps_tag in gps_data:
            sub_tag = GPSTAGS.get(gps_tag, gps_tag)
            gps[sub_tag] = gps_data[gps_tag]
        return self._convert_to_decimal_(gps)

    def _convert_to_decimal_(self, gps):
        """Convert degree, min, sec (dms) to decimal."""
        def dms_to_deg(dms, ref):
            degrees, minutes, seconds = dms
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        latitude, longitude, altitude = None, None, None
        if 'GPSLatitude' in gps and 'GPSLongitude' in gps:
            latitude = dms_to_deg(
                gps['GPSLatitude'], gps['GPSLatitudeRef']
            )
            longitude = dms_to_deg(
                gps['GPSLongitude'], gps['GPSLongitudeRef']
            )
            altitude = gps.get('GPSAltitude', None)

        return {
            "lat": latitude,
            "long": longitude,
            "alt": altitude
        }

    def get_base_gps(self):
        """Get the base gps coords of the images."""
        first_img = os.path.join(self.temp_dir, "processed_image_0.jpg")
        return self.get_gps(first_img)

    def get_img_size(self):
        """Return the image size."""
        return self.camera["img_size"]
