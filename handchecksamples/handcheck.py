import cv2
import mediapipe as mp

# --- MediaPipe Hands 初期化 ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 手検出器の設定
hands = mp_hands.Hands(
    static_image_mode=False,      # 動画用
    max_num_hands=2,              # 最大2手
    min_detection_confidence=0.7, # 検出信頼度
    min_tracking_confidence=0.7   # 追跡信頼度
)

# --- カメラ開始 ---
cap = cv2.VideoCapture(0)  # デバイスID 0 のカメラ

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    # BGR → RGB に変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手検出
    results = hands.process(frame_rgb)

    # 検出結果の描画
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 骨格を描画
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    # 画面表示
    cv2.imshow("Hand Skeleton", frame)

    # 'q' キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 後片付け ---
cap.release()
cv2.destroyAllWindows()
hands.close()
