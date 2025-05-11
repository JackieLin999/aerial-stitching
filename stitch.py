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

    # get the process images path
    processed_files = [
        os.path.join(wrapper.img_processor.temp_dir, f)
        for f in wrapper.photos
    ]
    N = len(processed_files)
    if N < 2:
        raise ValueError("Need at least two images to stitch.")

    if georef:
        mpp = wrapper.gps_info['ground_resolution']
        utm_offsets = { 
            f: wrapper.image_positions[f]['offset'] 
            for f in wrapper.photos
        }
    
    # now calculates all of the homographies
    homographies = [None] * N 
    # the first image will be the base image
    # there is NO need for a transformation, therefore an identity transformation
    homographies[0] = np.eye(3, dtype=np.float32)

    # now compute homographies for every image
    for i, fileName in enumerate(wrapper.photos):
        if i == 0:
            # inital homography alr set
            continue
        prev = processed_files[i - 1]
        curr = processed_files[i]

        # this is the homography through corners
        h_seg = wrapper.match(img_a=prev, img_b=curr)

        if georef:
            dx, dy = utm_offsets[wrapper.photos[i]]
            tx = dx / mpp
            ty = dy / mpp
            H_geo = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
            # combine homography: transform via corners then do little bit of shift with geo homography
            H_combined = H_geo @ H_seg
        else:
            H_combined = H_seg
        
        # to maintain that the homography is still "transforming" via the base image
        homographies[i] = homographies[i - 1] @ H_combined
    
    # compute canvas (thank you chat gpt)
    # essentially, we find the upper and lower left + right corners
    # and use them to create a canvas - canvas_size in our case
    all_corners = []
    for i, path in enumerate(processed_files):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        corners = np.array([[0,0], [w,0], [w,h], [0,h]], np.float32).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, homographies[i])
        all_corners.append(warped)
    all_corners = np.vstack(all_corners)
    min_xy = np.min(all_corners, axis=(0,1))
    max_xy = np.max(all_corners, axis=(0,1))
    offset = -min_xy.ravel()
    canvas_w = int(np.ceil(max_xy[0] - min_xy[0]))
    canvas_h = int(np.ceil(max_xy[1] - min_xy[1]))
    canvas_size = (canvas_w, canvas_h)

    # Initialize blender: blender will put all the images together
    blender = cv2.detail_MultiBandBlender() if hasattr(cv2, 'detail_MultiBandBlender') else cv2.detail.MultiBandBlender()
    blender.setNumBands(5)
    # Here's how the blender does it:
    # 1) transform each of the image
    # 2) put the image on its OWN canvas. the canvas size is the size of the result
    # so, a small image on a huge black background
    # 3) gather its left corners (0, 0) in our case because its alr transformed
    # also its mask, basically telling you where the image is located
    corners = []
    imgs = []
    masks = []

    # Warp and collect for blending
    for i, path in enumerate(processed_files):
        img = cv2.imread(path).astype(np.int16)
        H = homographies[i].copy()
        # this is to make sure all coordinates are positive
        # ex: the left most corner is (-1, -1). Offset will make it positive
        H[0,2] += offset[0]
        H[1,2] += offset[1]
        warped = cv2.warpPerspective(img, H, canvas_size)
        mask = cv2.warpPerspective(
            np.ones((img.shape[0], img.shape[1]), np.uint8),
            H, canvas_size
        )
        imgs.append(warped)
        masks.append(mask)
        corners.append((0,0))

    # Feed into blender
    blender.prepare(np.array(corners, np.int32))
    for img, mask in zip(imgs, masks):
        blender.feed(img, mask, (0,0))

    # Blend
    result, result_mask = blender.blend(None, None)
    mosaic = cv2.convertScaleAbs(result)

    # Save output
    cv2.imwrite(output_path, mosaic)
    print(f"Mosaic saved to {output_path}")