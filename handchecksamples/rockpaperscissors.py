import cv2
import mediapipe as mp
import random
import time

# === 画像パス（ここを後で変更してください） ===
IMG_ROCK = "rock.png"
IMG_PAPER = "paper.png"
IMG_SCISSORS = "scissors.png"

# 読み込み
img_rock = cv2.imread(IMG_ROCK)
img_paper = cv2.imread(IMG_PAPER)
img_scissors = cv2.imread(IMG_SCISSORS)

# サイズ調整（表示しやすい大きさへ）
img_rock = cv2.resize(img_rock, (200, 200))
img_paper = cv2.resize(img_paper, (200, 200))
img_scissors = cv2.resize(img_scissors, (200, 200))

# 画像辞書
hand_images = {
    "Rock": img_rock,
    "Paper": img_paper,
    "Scissors": img_scissors
}

# === Mediapipe 設定 ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

def get_finger_state(hand_landmarks):
    finger_state = []
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]

    # 親指（X軸）
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[mcp[0]].x:
        finger_state.append(1)
    else:
        finger_state.append(0)

    # 人差し指〜小指（Y軸）
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcp[i]].y:
            finger_state.append(1)
        else:
            finger_state.append(0)

    return finger_state

def get_hand_sign(finger_state):
    if finger_state == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_state == [1, 1, 1, 1, 1]:
        return "Paper"
    elif finger_state == [0, 1, 1, 0, 0]:
        return "Scissors"
    else:
        return None

def judge(player, cpu):
    if player == cpu:
        return "Draw"
    if (player == "Rock" and cpu == "Scissors") or \
       (player == "Paper" and cpu == "Rock") or \
       (player == "Scissors" and cpu == "Paper"):
        return "You Win!"
    return "You Lose..."

player_hand = None
cpu_hand = None
result_text = ""
countdown_active = False
countdown_start = 0
countdown_value = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 手の判定
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_state = get_finger_state(hand_landmarks)
            sign = get_hand_sign(finger_state)
            if sign:
                player_hand = sign

    key = cv2.waitKey(1) & 0xFF

    # SPACE で開始
    if key == ord(' '):
        countdown_active = True
        countdown_start = time.time()
        countdown_value = 3
        result_text = ""
        cpu_hand = None

    # Q または ESC で終了
    elif key == ord('q') or key == 27:
        break

    # カウントダウン
    if countdown_active:
        elapsed = time.time() - countdown_start
        countdown_value = 3 - int(elapsed)

        if countdown_value > 0:
            cv2.putText(frame, str(countdown_value), (250, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
        else:
            countdown_active = False
            if player_hand:
                cpu_hand = random.choice(["Rock", "Paper", "Scissors"])
                result_text = judge(player_hand, cpu_hand)

    # === 画像の描画 ===
    if player_hand in hand_images:
        frame[100:300, 10:210] = hand_images[player_hand]

    if cpu_hand in hand_images:
        frame[100:300, 430:630] = hand_images[cpu_hand]

    # 結果テキスト
    if result_text:
        cv2.putText(frame, result_text, (10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    # 説明
    cv2.putText(frame, "Press SPACE to Play JANKEN", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv2.imshow("Hand Janken Game", frame)

cap.release()
cv2.destroyAllWindows()