import argparse
import time
from modern_aerial_stitcher import ModernStitcher
from camera_info_extractor import CameraInfoFinder


def main():    
    parser = argparse.ArgumentParser(
        description='Aerial Image Stitching'
    )
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument('sensor_width')
    

if __name__ == "__main__":
    main()