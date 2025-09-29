import cv2
import os
import glob

input_folder = 'raw_images'
output_folder = 'processed_images'

os.makedirs(output_folder, exist_ok=True)

for img_path in glob.glob(f'{input_folder}/*.jpg'):
    img = cv2.imread(img_path)
    
    # Optional: Crop bottom 10% to remove props/legs
    h = img.shape[0]
    img = img[:int(h * 0.9), :]  
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Optional: Histogram equalization
    equalized = cv2.equalizeHist(blurred)
    
    # Save to output folder
    filename = os.path.basename(img_path)
    cv2.imwrite(f'{output_folder}/{filename}', equalized)
