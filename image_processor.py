# pylint: disable=no-member,unpacking-non-sequence
"""Preprocessor class."""
import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS
from PIL.ExifTags import TAGS


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

        self.images_dir = images_dir

        self.photos = [
            os.path.join(images_dir, photo)
            for photo in os.listdir(images_dir)
            if photo.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        if not self.photos:
            raise ValueError(f"No images found in {self.images_dir}")
        
        self.photos = sorted(
            [os.path.join(images_dir, photo) 
             for photo in os.listdir(images_dir) 
             if photo.lower().endswith(('.jpg', '.png', '.jpeg'))],
            key=self._get_image_timestamp
        )

        self._setup_camera()

    def _setup_camera(self):
        """Set up the camera intrinsics for cropping."""
        first_img_path = self.photos[0]
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
        processed_files = []  # Collect processed image paths
        for i, photo in enumerate(self.photos):
            img_path = photo
            img = cv2.imread(img_path)
            processed_photo = cv2.undistort(
                img, self.camera['mtx'],
                self.camera["distortion"], None, self.camera["new_mtx"]
            )
            processed_photo = processed_photo[y:y+h, x:x+w]
            processed_image_path = os.path.join(
                self.temp_dir,
                f"processed_image_{i}.png"
            )
            cv2.imwrite(processed_image_path, processed_photo)
            # Preserve EXIF data
            original_image = Image.open(img_path)
            exif_data = original_image.info.get("exif")
            if exif_data:
                processed_image = Image.open(processed_image_path)
                processed_image.save(processed_image_path, "JPEG", exif=exif_data)
            processed_files.append(processed_image_path)
        print("Image Processing successful")
        return processed_files  # Return paths to processed images

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
        try:
            # Parse latitude/longitude
            if 'GPSLatitude' in gps and 'GPSLatitudeRef' in gps:
                latitude = dms_to_deg(gps['GPSLatitude'], gps['GPSLatitudeRef'])
                
            if 'GPSLongitude' in gps and 'GPSLongitudeRef' in gps:
                longitude = dms_to_deg(gps['GPSLongitude'], gps['GPSLongitudeRef'])

            # Parse altitude with rational number handling
            if 'GPSAltitude' in gps:
                alt_data = gps['GPSAltitude']
                if isinstance(alt_data, tuple) and len(alt_data) == 2:
                    altitude = float(alt_data[0]) / float(alt_data[1])
                else:
                    altitude = float(alt_data)

                # Handle altitude reference (0 = above sea level, 1 = below)
                if 'GPSAltitudeRef' in gps and gps['GPSAltitudeRef'] == 1:
                    altitude = -abs(altitude)

        except (TypeError, ValueError, ZeroDivisionError) as e:
            print(f"GPS parsing warning: {str(e)}")
            # Preserve valid values if only altitude parsing failed
            altitude = None if isinstance(e, (TypeError, ZeroDivisionError)) else altitude

        return {
            "lat": latitude,
            "long": longitude,
            "alt": altitude
        }

    def get_base_gps(self):
        """Get the base gps coords of the images."""
        first_img = os.path.join(self.temp_dir, "processed_image_0.png")
        return self.get_gps(first_img)

    def get_img_size(self):
        """Return the image size."""
        return self.camera["img_size"]

    def _get_image_timestamp(self, img_path):
        image = Image.open(img_path)
        exif = image._getexif()
        if exif is None:
            return 0
        for tag, value in exif.items():
            if TAGS.get(tag) == 'DateTimeOriginal':
                return value
        return 0