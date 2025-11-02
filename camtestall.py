import cv2

print("=== カメラデバイスを順にテストします ===")

# 使えるバックエンドの候補
backends = [
    cv2.CAP_DSHOW,
    cv2.CAP_MSMF,
    cv2.CAP_VFW,
    cv2.CAP_ANY
]

for backend in backends:
    print(f"\n>>> テスト中: {backend}")
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("❌ 開けませんでした。")
        continue

    success, frame = cap.read()
    if not success:
        print("⚠️ カメラは開いたがフレームを取得できません。")
    else:
        print("✅ カメラ映像を取得できました。ウィンドウを表示します。")
        cv2.imshow(f"Camera Test - backend={backend}", frame)
        cv2.waitKey(3000)  # 3秒だけ表示

    cap.release()

cv2.destroyAllWindows()
print("\n=== テスト完了 ===")
