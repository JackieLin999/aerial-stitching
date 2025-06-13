import argparse
import time
from modern_aerial_stitcher import ModernStitcher
from camera_info_extractor import CameraInfoFinder


def main():
    # Step 1: Argument parsing
    parser = argparse.ArgumentParser(
        description='Aerial Image Stitching'
    )
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument('sensor_width')
    parser.add_argument('sensor_height')

    # Step 2: init camera infos extractor
    print("Initalize cam info extractor")
    args = parser.parse_args()
    cam_info_extractor = CameraInfoFinder(
        images_path=args.input_dir,
        sensor_width=float(args.sensor_width),
        sensor_height=float(args.sensor_height)
    )
    print("finished initalized cam info finder")
    print(f"in: {args.input_dir}")
    print(f"out: {args.output_path}")
    # extracting camera and image info
    print("Extracting camera information")
    try:
        principal_dist, altitude, camera_info, intrinsic_matrix, image_size = cam_info_extractor.get_intrinsic()
        print("Camera information extracted:")
        print(f"princial_dist: \n{principal_dist}")
        print(f"altitude: {altitude}")
        print(f"cam info: \n{camera_info}")
        print(f"intrinsic matrix: \n{intrinsic_matrix}")
        print(f"image size: {image_size}")
    except Exception as e:
        print("cam info extraction failed")
        print("Everything failed. lol.")
        print(e)
        return

    # step 3: initalize the stitcher
    usac_info = {
        'info': 'USAC_MAGSAC',
        'threshold': 0.7,
        'confidence': 0.9999,
        'max_iterations': 10000,
        'sigma': 3.0
    }
    # currently no way to find distortion coefficent
    camera_info= {
        'intrinsic_matrix': intrinsic_matrix,
        'distortion': None
    }
    stitcher = ModernStitcher(
        usac_info=usac_info,
        camera_info=camera_info,
        img_size=image_size,
        imgs_path=args.input_dir,
        output_path=args.output_path
    )
    stitcher.stitch_all()


if __name__ == "__main__":
    main()