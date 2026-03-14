import cv2

# Open camera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    # Press 's' to save image
    if key == ord('s'):
        count += 1
        cv2.imwrite(f"Calibration images/Photo {count}.jpg", frame)
        print("Photo saved!")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()