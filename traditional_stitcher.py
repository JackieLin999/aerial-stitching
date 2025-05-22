"""Stitches images via traditional methods like sift and orb."""
import os
import cv2
import numpy as np
from wrapper import Wrapper


class TraditionalAerialStitcher:
    """Class for stitching aerial images."""

    def __init__(
        self,
        input_dir: str,
        output_path: str,
        focal_length: float = 8331,
        principal_x: float = 2000,
        principal_y: float = 1500,
        nfeats: int = 20000,
        sensor_width: float = 0.0096,
        georef: bool = True
    ):
        """Init the traditional aerial stitcher."""
        self.input_dir = input_dir
        self.output_path = output_path

        self.camera = {
            "focal_length": focal_length,
            "sensor_width": sensor_width,
        }

        self.inputs = {
            "georef": georef,
            "nfeats": nfeats
        }

        self.principal_x = principal_x
        self.principal_y = principal_y

        # Initialize the wrapper
        self.wrapper = Wrapper(
            input_dir=self.input_dir,
            focal_length=self.camera['focal_length'],
            principal_x=self.principal_x,
            principal_y=self.principal_y,
            nfeats=self.inputs['nfeats'],
            sensor_width=self.camera['sensor_width'],
            orb=False
        )
        print("Wrapper class initialized")

    def stitch_images(self):
        """Stitch all of the images."""
        if self.inputs['georef']:
            print("Preparing geo infos")
            mpp = self.wrapper.gps_info['ground_resolution']
            print(f"The mpp: {mpp}")
            utm_offsets = {
                f: self.wrapper.image_positions[f]['offset']
                for f in self.wrapper.photos
            }
        else:
            mpp = None
            utm_offsets = None

        clusters = self.wrapper.clusters
        cluster_output_dir = os.path.join(os.getcwd(), self.output_path)
        os.makedirs(cluster_output_dir, exist_ok=True)

        for cluster_index, cluster in enumerate(clusters):
            print(f"Processing cluster {cluster_index + 1}/{len(clusters)}...")
            homographies = []
            print("Start calculating homography")

            cluster_size = len(cluster)
            H_base = np.eye(3, dtype=np.float32)
            cluster_homo = [H_base]
            for i in range(1, cluster_size):
                prev_img = cluster[i-1]
                curr_img = cluster[i]

                h_seg = self.wrapper.match(prev_img, curr_img)
                if h_seg is None:
                    print("Insufficient matches, using identity matrix.")
                    h_seg = np.eye(3, dtype=np.float32)

                h_seg_inv = np.linalg.inv(h_seg)
                if self.inputs['georef']:
                    base_img = cluster[0]
                    base_dx, base_dy = utm_offsets[base_img]

                    curr_dx, curr_dy = utm_offsets[curr_img]
                    tx = (curr_dx - base_dx) / mpp
                    ty = (curr_dy - base_dy) / mpp

                    h_geo = np.array(
                        [
                            [1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]
                        ],
                        dtype=np.float32
                    )
                    h_combined = h_geo @ h_seg_inv
                else:
                    h_combined = h_seg_inv

                h_cum = cluster_homo[i-1] @ h_combined
                cluster_homo.append(h_cum)

            print("Calculating canvas size")
            all_corners = []
            cluster_images = [cv2.imread(img_path) for img_path in cluster]
            for i, img in enumerate(cluster_images):
                h, w = img.shape[:2]
                corners = np.array(
                    [
                        [0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]
                    ], dtype=np.float32
                ).reshape(-1, 1, 2)
                warped = cv2.perspectiveTransform(corners, cluster_homo[i])
                all_corners.append(warped)

            all_corners = np.vstack(all_corners)
            min_xy = np.min(all_corners, axis=(0, 1))
            max_xy = np.max(all_corners, axis=(0, 1))
            offset = -min_xy.ravel()
            canvas_size = (
                int(np.ceil(max_xy[0] - min_xy[0])),
                int(np.ceil(max_xy[1] - min_xy[1]))
            )
            print("Finished calculating canvas size")

            cluster_blender = cv2.detail_MultiBandBlender()
            cluster_blender.prepare((0, 0, canvas_size[0], canvas_size[1]))

            for img, h in zip(cluster_images, cluster_homo):
                h_translate = np.array(
                    [[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]],
                    dtype=np.float32
                )
                warped_img = cv2.warpPerspective(
                    img,
                    h_translate @ h, canvas_size
                )
                mask = np.ones(
                    (
                        img.shape[0],
                        img.shape[1]
                    ),
                    dtype=np.uint8
                ) * 255
                warped_mask = cv2.warpPerspective(
                    mask,
                    h_translate @ h,
                    canvas_size
                )
                cluster_blender.feed(warped_img, warped_mask, (0, 0))

            result, result_mask = cluster_blender.blend(None, None)
            stitched_image = cv2.convertScaleAbs(result)
            cluster_output_path = os.path.join(
                cluster_output_dir,
                f"cluster_{cluster_index + 1}.png"
            )
            cv2.imwrite(cluster_output_path, stitched_image)
            print(
                f"Cluster {cluster_index + 1} stitched and saved to "
                f"{cluster_output_path}."
            )

        print("All clusters processed and saved.")
