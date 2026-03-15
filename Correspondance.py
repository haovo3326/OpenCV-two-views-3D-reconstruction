import cv2
import numpy as np
import torch
import kornia.feature as KF

img1 = cv2.imread("Object images/Image 1.png")
img2 = cv2.imread("Object images/Image 2.png")

def get_correspondence():
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one or both images.")

    # LoFTR expects grayscale images
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Resize
    scale = 1
    gray1_small = cv2.resize(gray1, None, fx = scale, fy = scale)
    gray2_small = cv2.resize(gray2, None, fx=scale, fy=scale)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor with shape [1, 1, H, W]
    timg1 = torch.from_numpy(gray1_small).float() / 255.0
    timg2 = torch.from_numpy(gray2_small).float() / 255.0

    timg1 = timg1.unsqueeze(0).unsqueeze(0).to(device)
    timg2 = timg2.unsqueeze(0).unsqueeze(0).to(device)

    # Load pretrained LoFTR
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    with torch.no_grad():
        correspondences = matcher({
            "image0": timg1,
            "image1": timg2
        })

    # Extract matched points
    pts1 = correspondences["keypoints0"].cpu().numpy().astype(np.float32)
    pts2 = correspondences["keypoints1"].cpu().numpy().astype(np.float32)
    conf = correspondences["confidence"].cpu().numpy()

    # Keep only stronger matches
    mask = conf > 0.3
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    conf = conf[mask]

    # Keep only top-N
    max_matches = 1000
    idx = np.argsort(-conf)[:max_matches]
    pts1 = pts1[idx]
    pts2 = pts2[idx]

    # scale points back to original image size
    pts1[:, 0] /= scale
    pts1[:, 1] /= scale
    pts2[:, 0] /= scale
    pts2[:, 1] /= scale

    return pts1, pts2

# pts1, pts2 = get_correspondence()
# print(f"Number of correspondences: {len(pts1)}")
#
# # copy images for drawing
# draw1 = img1.copy()
# draw2 = img2.copy()
#
# # draw points
# for (x1, y1), (x2, y2) in zip(pts1, pts2):
#     cv2.circle(draw1, (int(x1), int(y1)), 3, (0,255,0), -1)
#     cv2.circle(draw2, (int(x2), int(y2)), 3, (0,255,0), -1)
#
# # create single canvas
# canvas = cv2.hconcat([draw1, draw2])
#
# # width offset for second image
# offset = img1.shape[1]
# # draw lines between matches
# for (x1, y1), (x2, y2) in zip(pts1, pts2):
#     pt1 = (int(x1), int(y1))
#     pt2 = (int(x2 + offset), int(y2))
#     cv2.line(canvas, pt1, pt2, (255,0,0), 1)
#
# # scale canvas for display
# display_scale = 0.5
# small_canvas = cv2.resize(canvas, None, fx=display_scale, fy=display_scale)
#
# cv2.imshow("Matches", small_canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
