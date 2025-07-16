# Matching
```
    parser.add_argument('--megadepth_root_path', type=str, default='/mnt/usb/zdzdata/MegaDepth_v1',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/mnt/usb/zdzdata/coco_20k/coco_20k',
                        help='Path to the synthetic dataset root directory.')
```

```
            TRAIN_BASE_PATH = f"{megadepth_root_path}/index"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/train/phoenix/S6/zl548/MegaDepth_v1"
            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

```

![image](https://github.com/user-attachments/assets/75ec174f-9966-428d-aff1-a9c58c245a9a)

![image](https://github.com/user-attachments/assets/11020ba9-fc00-4f6c-9a72-5bf383c7c6be)

==文章十分重要，在【13】中讲述了如何进行轻量化设计，以及正文中都有

原始test代码
```
import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from modules.xfeat import XFeat

xfeat = XFeat()

#Load some example images
im1 = cv2.imread('Color_20240607_164314_No6_e400_g0_8bit.png')
im2 = cv2.imread('IR_20240607_164314_No6_8bit.png')



def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

#Use out-of-the-box function for extraction + coarse-to-fine matching
mkpts_0, mkpts_1 = xfeat.match_xfeat_star(im1, im2, top_k = 8000)

# canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
# plt.figure(figsize=(12,12))
# plt.imshow(canvas[..., ::-1]), plt.show()

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
# 保存图像到文件而不是显示
cv2.imwrite('result_image.png', canvas)
print("Image saved as result_image.png")

```


新的自己写的匹配
```
import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from modules.xfeat import XFeat

xfeat = XFeat()

#Load some example images
im1 = cv2.imread('2767b8c57a36462959c2961fd75b4698.jpg')
im2 = cv2.imread('cd5418b3048e1842870bcb8f80e15d8d.jpg')


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches, H


# Use out-of-the-box function for extraction + coarse-to-fine matching
mkpts_0, mkpts_1 = xfeat.match_xfeat_star(im1, im2, top_k = 8000)

# Get matches and homography matrix
canvas, H = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)

# Apply the homography to im1 to warp it into im2's perspective
h, w = im2.shape[:2]  # Get the size of the second image
warped_im1 = cv2.warpPerspective(im1, H, (w, h))

# Optionally: Show the warped image
plt.figure(figsize=(12, 12))
plt.imshow(warped_im1[..., ::-1])  # Convert BGR to RGB for display
plt.title('Warped Image')
plt.show()

# Save the result
cv2.imwrite('warped_image.png', warped_im1)
print("Warped image saved as warped_image.png")

```
