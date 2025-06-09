"""Main function to run the stitching process."""
import argparse
import time
from traditional_stitcher import TraditionalAerialStitcher


def main():
    """Stitch the images and follow the commands user pointed out."""
    parser = argparse.ArgumentParser(
        description='Aerial Image Stitching w/ optional georeference'
    )
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument(
        '--no-georef',
        action='store_false',
        dest='georef',
        help='Disable GPS georeferencing'
    )
    args = parser.parse_args()

    start_time = time.time()
    stitcher = TraditionalAerialStitcher(
        input_dir=args.input_dir,
        output_path=args.output_path,
        georef=args.georef
    )
    stitcher.stitch_images()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()
