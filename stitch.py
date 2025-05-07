import cv2
import os
import numpy as np

"""preprocessing"""
focal_length = 4800
principal_x = 2254
principal_y = 2048
mtx = np.array([
    [focal_length, 0, principal_x],
    [0, focal_length, principal_y],
    [0, 0, 1]
])
distortion_coeff = np.zeros(5)

images_dir = os.path.join(os.getcwd(), "images/test")
photos = [photo for photo in os.listdir(images_dir) if photo.lower().endswith(('.jpg', '.png', '.jpeg'))]
if photos:
    first_img_path = os.path.join(images_dir, photos[0])
    img = cv2.imread(first_img_path)
    h, w = img.shape[:2]
    img_size = (w, h)
else:
    print("empty directy at ", images_dir)

new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion_coeff, (w, h), 1, (w, h))

temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)
x, y, w, h = roi
for i, photo in enumerate(photos):
    img_path = os.path.join(images_dir, photo)
    img = cv2.imread(img_path)
    processed_photo = cv2.undistort(img, mtx, distortion_coeff, None, new_mtx)
    #crop via regin of interest
    processed_photo = processed_photo[y:y+h, x:x+w]
    processed_image_path = os.path.join(temp_dir, f"processed_image_{i}.jpg")
    cv2.imwrite(processed_image_path, processed_photo)
