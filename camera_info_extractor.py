"""Finds the necessary info on the camera."""
import os
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.ExifTags import GPSTAGS

class CameraInfoFinder:
    """A class for extracting the intrinsic info."""
    
    def __init__(self, images_path, sensor_width, sensor_height):
        """Init the camera info finder class."""
        if not isinstance(sensor_width, int):
            raise TypeError("sensor_width must be an integer.")
        if sensor_width <= 0:
            raise ValueError("sensor_width must be a positive integer.")
        self.images_path = images_path
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

    def get_intrinsic(self):
        """Extract the intrisic info of the camera."""
        if not os.path.exists(self.images_path):
            # warn user 
            print("PATH GIVEN IS INVALIED")
            print(f"invalid path: {self.images_path}")
            print("terminating process")
            raise FileNotFoundError(f"The specified path does not exist: {self.images_path}")
        
        image = self._get_first_img()
        width, height = image.size
        principal_x = width // 2
        principal_y = height // 2
        principal_dist = {
            "principal_x": principal_x,
            "principal_y": principal_y
        }

        exif_data = image._getexif()
        altitude = self._extract_altitude(exif_data=exif_data)

        focal_length = self._extract_focal_length(exif_data=exif_data)
        camera_info = {
            "focal_length": focal_length,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height
        }

        intrinsic_matrix = self.create_intrinsic_matrix(
            focal=focal_length,
            principal_dist=principal_dist
        )

        return principal_dist, altitude, camera_info, intrinsic_matrix

    def _extract_focal_length(self, exif_data):
        """Extract the focal length from the image."""
        if exif_data:
            focal_length_tag = 37386
            if focal_length_tag in exif_data:
                focal_value = exif_data[focal_length_tag]
                if isinstance(focal_value, tuple):
                    focal_length = focal_value[0] / focal_value[1]
                else:
                    focal_length = focal_value
        return focal_length

    def _extract_altitude(self, exif_data):
        """Extract the altitude from the image."""
        if exif_data:
            gps_info = exif_data.get(34853)
            if gps_info:
                gps_data = {}
                for key, value in gps_info.items():
                    tag_name = GPSTAGS.get(key, key)
                    gps_data[tag_name] = value

                # Extract altitude (if available)
                if 'GPSAltitude' in gps_data:
                    numerator, denominator = gps_data['GPSAltitude']
                    altitude = numerator / denominator
                    # Handle altitude reference (0 = above sea level, 1 = below)
                    altitude_ref = gps_data.get('GPSAltitudeRef', 0)
                    if altitude_ref == 1:
                        altitude = -altitude
        return altitude

    def _get_first_img(self):
        """Get the first image."""
        for image in os.listdir(self.images_path):
            return Image.open(image)
    
    def create_intrinsic_matrix(self, focal, principal_dist):
        principal_x = principal_dist['principal_x']
        principal_y = principal_dist['principal_y']
        image_width = principal_x * 2
        image_height = principal_y * 2
        focal_x = (focal * image_width) // self.sensor_width
        focal_y = (focal * image_height) // self.sensor_height
        return np.array([
            [focal_x, 0, principal_x],
            [0, focal_y, principal_y],
            [0, 0, 1]
        ])
