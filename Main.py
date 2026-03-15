import Calibration
import Correspondance
import cv2
import numpy as np
import matplotlib.pyplot as plt

K = Calibration.calibrate()
pts1, pts2 = Correspondance.get_correspondence()

E, m = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=m)

# Keep only inlier points
pts1_in = pts1[mask.ravel() > 0]
pts2_in = pts2[mask.ravel() > 0]

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

points4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
points3D = (points4D[:3] / points4D[3]).T

# Depth in camera 1
Z1 = points3D[:, 2]

# Depth in camera 2
points_cam2 = (R @ points3D.T + t).T
Z2 = points_cam2[:, 2]

valid = (Z1 > 0) & (Z2 > 0)
points3D = points3D[valid]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[:,0], points3D[:,1], points3D[:,2], s=3)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()


