import cv2
import os

# === CONFIGURATION ===
camera_source = 0  # 0 for USB/CSI cam; replace with RTSP URL for drone feed
num_images_to_capture = 10
raw_folder = 'raw_images'
processed_folder = 'processed_images'

# === SETUP ===
os.makedirs(raw_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(camera_source)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting image capture...")

for i in range(num_images_to_capture):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # === SAVE RAW IMAGE ===
    raw_filename = os.path.join(raw_folder, f'image_{i:03d}.jpg')
    cv2.imwrite(raw_filename, frame)
    print(f"[{i+1}/{num_images_to_capture}] Saved raw: {raw_filename}")

    # === PREPROCESSING ===
    # Optional: Crop bottom 10%
    h = frame.shape[0]
    cropped = frame[:int(h * 0.9), :]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Histogram equalization
    equalized = cv2.equalizeHist(blurred)

    # === SAVE PROCESSED IMAGE ===
    processed_filename = os.path.join(processed_folder, f'image_{i:03d}.jpg')
    cv2.imwrite(processed_filename, equalized)
    print(f"[{i+1}/{num_images_to_capture}] Saved processed: {processed_filename}")

cap.release()
print("Image capture and preprocessing complete.")
