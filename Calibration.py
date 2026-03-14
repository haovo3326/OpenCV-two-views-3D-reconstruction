import cv2

img = cv2.imread("Calibration images/Photo 1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pattern_size = (4, 4)

ret, corners = cv2.findChessboardCornersSB(gray, pattern_size)
points = corners.reshape(-1, 2)
print(points)

if ret:
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    cv2.imshow("Corners SB", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Checkerboard not found")