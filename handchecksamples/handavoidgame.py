import cv2
import mediapipe as mp
import random
import time

# Mediapipe Hands 設定
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# ウィンドウサイズ
WIDTH = 640
HEIGHT = 480

# プレイヤー設定
PLAYER_RADIUS = 25
player_x = WIDTH // 2
player_y = HEIGHT - 70

# 敵の設定
ENEMY_SIZE = 50
enemy_list = []
enemy_speed = 3
last_spawn_time = 0

# ゲーム状態
game_over = False
score = 0
start_time = time.time()

# 敵生成関数
def spawn_enemy():
    x = random.randint(0, WIDTH - ENEMY_SIZE)
    return [x, -ENEMY_SIZE]

# ゲームリセット
def reset_game():
    global enemy_list, enemy_speed, game_over, start_time, score
    enemy_list = []
    enemy_speed = 3
    game_over = False
    score = 0
    start_time = time.time()

# カメラ開始
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 左右反転（鏡風）
    frame = cv2.flip(frame, 1)

    # 黒背景
    game_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    game_frame[:] = 0
    game_frame = cv2.cvtColor(game_frame, cv2.COLOR_GRAY2BGR)

    # Mediapipe 処理
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not game_over:
        # 手の位置取得
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            x = lm.landmark[mp_hands.HandLandmark.WRIST].x
            player_x = int(x * WIDTH)

        # 敵の生成（0.8秒ごと）
        if time.time() - last_spawn_time > 0.8:
            enemy_list.append(spawn_enemy())
            last_spawn_time = time.time()

        # 敵の移動
        for enemy in enemy_list:
            enemy[1] = int(enemy[1] + enemy_speed)

        # 画面外の敵を削除
        enemy_list = [e for e in enemy_list if e[1] < HEIGHT]

        # スコア（生存時間）
        score = int(time.time() - start_time)

        # 敵スピード上昇
        enemy_speed = 3 + (score * 0.2)

        # 当たり判定
        for ex, ey in enemy_list:
            if (ex < player_x < ex + ENEMY_SIZE) and (ey < player_y < ey + ENEMY_SIZE):
                game_over = True

    # プレイヤー描画（白い円）
    cv2.circle(game_frame, (int(player_x), int(player_y)), PLAYER_RADIUS, (255, 255, 255), -1)

    # 敵描画（赤い四角） ※整数化済み
    for ex, ey in enemy_list:
        cv2.rectangle(
            game_frame,
            (int(ex), int(ey)),
            (int(ex + ENEMY_SIZE), int(ey + ENEMY_SIZE)),
            (0, 0, 255),
            -1
        )

    # スコア表示
    cv2.putText(game_frame, f"Score: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # ゲームオーバー表示
    if game_over:
        cv2.putText(game_frame, "GAME OVER", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
        cv2.putText(game_frame, "Press SPACE to Restart", (110, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # 操作説明
    cv2.putText(game_frame, "Press ESC or Q to Quit", (10, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    # 表示
    cv2.imshow("Hand Avoid Game", game_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    if key == ord('q'):
        break
    if key == ord(' ') and game_over:
        reset_game()

# 終了処理
cap.release()
cv2.destroyAllWindows()
