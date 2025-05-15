import os
import cv2
import numpy as np
from wrapper import Wrapper

def stitch_aerial_images(
    input_dir: str,
    output_path: str,
    focal_length: float = 4800,
    principal_x: float = 2254,
    principal_y: float = 2048,
    nfeats: int = 5000,
    sensor_width: float = 0.01127,
    georef: bool = True
):
    """Stitch all of the images"""
    wrapper = Wrapper(
        input_dir=input_dir,
        focal_length=focal_length,
        principal_x=principal_x,
        principal_y=principal_y,
        nfeats=nfeats,
        sensor_width=sensor_width
    )
    print("wrapper class initalized")
    # get the process images path
    clusters = wrapper.clusters
    cluster_output_dir = os.path.join(os.getcwd(), output_path)
    os.makedirs(cluster_output_dir, exist_ok=True)
    
    mpp = wrapper.gps_info['ground_resolution'] if georef else None
    for cluster_index, cluster in enumerate(clusters):
        print(f"Processing cluster {cluster_index + 1}/{len(clusters)}...")
        homographies = []

        # Get base image's UTM offset for this cluster
        # if georef:
        #     base_img = cluster[0]
        #     base_x, base_y = utm_offsets[base_img]
        #     H_global = np.array([
        #         [1, 0, base_x/mpp],
        #         [0, 1, base_y/mpp],
        #         [0, 0, 1]
        #     ], dtype=np.float32)
        # else:
        #     H_global = np.eye(3, dtype=np.float32)

        # computing homographies for each clusters
        print("start calculating homography")
        cluster_size = len(cluster)
        # init w identity matrix for the 1st image or ur base img for that cluster
        cluster_homo = [np.eye(3, dtype=np.float32)]
        for i in range(1, cluster_size):
            prev_img = cluster[i-1]
            curr_img = cluster[i]

            h_seg = wrapper.match(prev_img, curr_img)
            if h_seg is None:
                # lol just forced it
                print("fuck it")
                h_seg = np.eye(3, dtype=np.float32)
            h_seg_inv = np.linalg.inv(h_seg)
            # geo ref stuff
            if georef:
                dx, dy = utm_offsets[wrapper.photos[i]]
                tx = dx / mpp
                ty = dy / mpp
                h_geo = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
                # combine homography: transform via corners then do little bit of shift with geo homography
                h_combined = h_geo @ h_seg_inv
            else:
                h_combined = h_seg_inv
            
            h_cum = cluster_homo[i-1] @ h_combined
            cluster_homo.append(h_cum)

        # canvas size
        print("calculating canvas size")
        all_corners = []
        cluster_images = [cv2.imread(img_path) for img_path in cluster]
        for i, img in enumerate(cluster_images):
            h, w = img.shape[:2]
            corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32).reshape(-1,1,2)
            warped = cv2.perspectiveTransform(corners, cluster_homo[i])
            all_corners.append(warped)
        
        all_corners = np.vstack(all_corners)
        min_xy = np.min(all_corners, axis=(0,1))
        max_xy = np.max(all_corners, axis=(0,1))
        offset = -min_xy.ravel()
        canvas_size = (int(np.ceil(max_xy[0] - min_xy[0])), 
                    int(np.ceil(max_xy[1] - min_xy[1])))
        print("Finished calculating canvas size")

        cluster_blender = cv2.detail_MultiBandBlender()
        cluster_blender.prepare((0, 0, canvas_size[0], canvas_size[1]))

        # Feed into blender
        for img, h in zip(cluster_images, cluster_homo):
            h_translate = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], dtype=np.float32)
            warped_img = cv2.warpPerspective(img, h_translate @ h, canvas_size)
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, h_translate @ h, canvas_size)
            cluster_blender.feed(warped_img, warped_mask, (0, 0))

        # save it
        result, result_mask = cluster_blender.blend(None, None)
        stitched_image = cv2.convertScaleAbs(result)
        cluster_output_path = os.path.join(cluster_output_dir, f"cluster_{cluster_index + 1}.jpg")
        cv2.imwrite(cluster_output_path, stitched_image)
        print(f"Cluster {cluster_index + 1} stitched and saved to {cluster_output_path}.")

    print("All clusters processed and saved.")
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aerial Image Stitching w/ optional georeference')
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument('--no-georef', action='store_false', dest='georef', help='Disable GPS georeferencing')
    args = parser.parse_args()
    stitch_aerial_images(args.input_dir, args.output_path, georef=args.georef)