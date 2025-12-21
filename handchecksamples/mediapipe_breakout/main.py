import cv2
import mediapipe as mp
import numpy as np

from stages import STAGE_1

# =====================
# 画面設定
# =====================
WIDTH, HEIGHT = 800, 600

# =====================
# パドル
# =====================
PADDLE_W = 120
PADDLE_H = 15
PADDLE_Y = HEIGHT - 40

# =====================
# ボール
# =====================
BALL_R = 8
BALL_SPEED = 5

ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_vx = BALL_SPEED
ball_vy = -BALL_SPEED

# =====================
# MediaPipe
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =====================
# ステージAPI
# =====================
def load_stage(stage):
    blocks = []
    bw, bh = stage["block_size"]
    pad = stage["padding"]
    ox, oy = stage["offset"]

    for ry, row in enumerate(stage["layout"]):
        for rx, ch in enumerate(row):
            info = stage["block_types"].get(ch)
            if info is None:
                continue

            blocks.append({
                "x": ox + rx * (bw + pad),
                "y": oy + ry * (bh + pad),
                "w": bw,
                "h": bh,
                "hp": info["hp"],
                "color": info["color"],
                "effect": info.get("effect")
            })
    return blocks

# =====================
# ボール効果API（核心）
# =====================
def apply_block_effect(effect):
    global ball_vx, ball_vy

    if effect == "speed_up":
        ball_vx *= 1.2
        ball_vy *= 1.2

    elif effect == "slow_down":
        ball_vx *= 0.8
        ball_vy *= 0.8

# =====================
# 初期化
# =====================
blocks = load_stage(STAGE_1)
paddle_x = WIDTH // 2 - PADDLE_W // 2

# =====================
# カメラ
# =====================
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Block Breaker")
cv2.resizeWindow("Hand Block Breaker", WIDTH, HEIGHT)

# =====================
# メインループ
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # ---- 手入力 → パドル
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        wrist_x = lm[mp_hands.HandLandmark.WRIST].x
        paddle_x = int(wrist_x * WIDTH - PADDLE_W / 2)
        paddle_x = max(0, min(WIDTH - PADDLE_W, paddle_x))

    # ---- ボール移動
    ball_x += ball_vx
    ball_y += ball_vy

    # 壁
    if ball_x < BALL_R or ball_x > WIDTH - BALL_R:
        ball_vx *= -1
    if ball_y < BALL_R:
        ball_vy *= -1

    # パドル
    if (PADDLE_Y < ball_y + BALL_R < PADDLE_Y + PADDLE_H and
        paddle_x < ball_x < paddle_x + PADDLE_W):
        ball_vy *= -1
        ball_y = PADDLE_Y - BALL_R

    # ブロック衝突
    for block in blocks[:]:
        if (block["x"] < ball_x < block["x"] + block["w"] and
            block["y"] < ball_y < block["y"] + block["h"]):

            block["hp"] -= 1
            ball_vy *= -1

            if block.get("effect"):
                apply_block_effect(block["effect"])

            if block["hp"] <= 0:
                blocks.remove(block)
            break

    # ---- 描画
    cv2.rectangle(
        frame,
        (paddle_x, PADDLE_Y),
        (paddle_x + PADDLE_W, PADDLE_Y + PADDLE_H),
        (255, 255, 255),
        -1
    )

    cv2.circle(
        frame,
        (int(ball_x), int(ball_y)),
        BALL_R,
        (0, 255, 255),
        -1
    )

    for block in blocks:
        cv2.rectangle(
            frame,
            (block["x"], block["y"]),
            (block["x"] + block["w"], block["y"] + block["h"]),
            block["color"],
            -1
        )

    cv2.imshow("Hand Block Breaker", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================
# 終了処理
# =====================
cap.release()
cv2.destroyAllWindows()
hands.close()
