import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("カメラ映像を取得できません。")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow('Camera Test (No MediaPipe)', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
