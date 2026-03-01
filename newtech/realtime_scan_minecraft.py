import os
import time
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


WIDTH = 960
HEIGHT = 720
WINDOW_NAME = "Realtime Scan -> Minecraft"

PINCH_RATIO = 0.35
MIN_TRACE_DISTANCE_PX = 6.0
PIXELS_PER_BLOCK = 10.0
SCALE_MIN = 0.3
SCALE_MAX = 4.0
SCALE_SENSITIVITY = 3.0

MC_REMOTE_HOST = os.environ.get("MC_REMOTE_HOST", "mc-remote.xgames.jp")
MC_REMOTE_PORT = int(os.environ.get("MC_REMOTE_PORT", "25575"))
MC_REMOTE_PLAYER_NAME = os.environ.get(
    "MC_REMOTE_PLAYER_NAME",
    os.environ.get("MINECRAFT_PLAYER_NAME", os.environ.get("USERNAME", "")),
)
MC_REMOTE_ORIGIN_X = int(os.environ.get("MC_REMOTE_ORIGIN_X", "1000"))
MC_REMOTE_ORIGIN_Y = int(os.environ.get("MC_REMOTE_ORIGIN_Y", "100"))
MC_REMOTE_ORIGIN_Z = int(os.environ.get("MC_REMOTE_ORIGIN_Z", "1000"))
MC_REMOTE_BLOCK = os.environ.get("MC_REMOTE_BLOCK", "lime_concrete")


def finger_states(hand_landmarks) -> List[int]:
    lm = hand_landmarks.landmark

    def up(tip_idx: int, pip_idx: int) -> int:
        return int(lm[tip_idx].y < lm[pip_idx].y)

    thumb_open = int(abs(lm[4].x - lm[2].x) > 0.03)
    return [
        thumb_open,
        up(8, 6),
        up(12, 10),
        up(16, 14),
        up(20, 18),
    ]


def is_open_hand(state: Sequence[int]) -> bool:
    return sum(state) >= 4


def is_fist(state: Sequence[int]) -> bool:
    return sum(state) <= 1


def is_thumb_up(hand_landmarks, state: Sequence[int]) -> bool:
    lm = hand_landmarks.landmark
    others_folded = state[1:] == [0, 0, 0, 0]
    thumb_upward = lm[4].y < lm[3].y < lm[2].y
    return bool(others_folded and thumb_upward)


def is_pinch(hand_landmarks) -> bool:
    lm = hand_landmarks.landmark
    pinch_dist = dist((lm[4].x, lm[4].y), (lm[8].x, lm[8].y))
    hand_scale = dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y))
    if hand_scale < 1e-6:
        return False
    return pinch_dist < hand_scale * PINCH_RATIO


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def scale_points(points: Sequence[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    if not points:
        return []
    arr = np.array(points, dtype=np.float32)
    center = arr.mean(axis=0)
    scaled = (arr - center) * scale + center
    return [(float(p[0]), float(p[1])) for p in scaled]


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return pts


def to_relative_blocks(points: Sequence[Tuple[float, float]]) -> List[Tuple[int, int]]:
    if len(points) < 2:
        return []

    block_pts = []
    for x, y in points:
        bx = int(round((x - WIDTH / 2.0) / PIXELS_PER_BLOCK))
        by = int(round((HEIGHT / 2.0 - y) / PIXELS_PER_BLOCK))
        block_pts.append((bx, by))

    traced = []
    for i in range(1, len(block_pts)):
        traced.extend(bresenham_line(block_pts[i - 1][0], block_pts[i - 1][1], block_pts[i][0], block_pts[i][1]))

    if dist(points[0], points[-1]) < 40.0:
        traced.extend(bresenham_line(block_pts[-1][0], block_pts[-1][1], block_pts[0][0], block_pts[0][1]))

    unique = sorted(set(traced))
    return unique


def draw_polyline(frame, points: Sequence[Tuple[float, float]], color, thickness: int = 2) -> None:
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        p0 = (int(points[i - 1][0]), int(points[i - 1][1]))
        p1 = (int(points[i][0]), int(points[i][1]))
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)


def put_help(
    frame,
    scale: float,
    point_count: int,
    block_count: int,
    connected: bool,
    status_text: str,
) -> None:
    lines = [
        f"MC Remote: {'Connected' if connected else 'Disconnected'}",
        "Pinch(thumb+index): trace contour",
        "Thumb up + move up/down: scale adjust",
        "Open hand: rebuild in Minecraft",
        "Fist: clear contour + clear blocks",
        "B: rebuild now | C: clear blocks | ESC: exit",
        f"Scale: {scale:.2f} | Points: {point_count} | Blocks: {block_count}",
        f"Status: {status_text}",
    ]

    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
        y += 28


class MCShapeWriter:
    def __init__(self):
        self.mc = None
        self.block_id = None
        self.active = False
        self.placed = set()

    def connect(self) -> None:
        if not MC_REMOTE_PLAYER_NAME:
            print("[MC_REMOTE] player name is empty. Set MC_REMOTE_PLAYER_NAME.")
            return

        try:
            from mc_remote.minecraft import Minecraft
            from mc_remote import block_id

            self.mc = Minecraft.create(address=MC_REMOTE_HOST, port=MC_REMOTE_PORT)
            self.mc.setPlayer(MC_REMOTE_PLAYER_NAME, MC_REMOTE_ORIGIN_X, 0, MC_REMOTE_ORIGIN_Z)
            self.block_id = block_id
            self.active = True
            self.mc.postToChat("[Realtime Scan] Ready")
            print(
                "[MC_REMOTE] connected "
                f"host={MC_REMOTE_HOST}:{MC_REMOTE_PORT} player={MC_REMOTE_PLAYER_NAME} "
                f"origin=({MC_REMOTE_ORIGIN_X},{MC_REMOTE_ORIGIN_Y},{MC_REMOTE_ORIGIN_Z})"
            )
        except ModuleNotFoundError:
            print("[MC_REMOTE] minecraft-remote-api is not installed.")
            print("[MC_REMOTE] Install with: python -m pip install minecraft-remote-api")
        except Exception as exc:
            print(f"[MC_REMOTE] connection failed: {exc}")

    def resolve_block(self, name: str):
        key = name.upper()
        return getattr(self.block_id, key, name)

    def clear(self) -> None:
        if not self.active:
            return
        air = self.resolve_block("air")
        for x, y, z in list(self.placed):
            self.mc.setBlock(x, y, z, air)
        self.placed.clear()

    def rebuild_shape(self, relative_blocks: Sequence[Tuple[int, int]]) -> int:
        if not self.active:
            return 0

        self.clear()
        block_name = self.resolve_block(MC_REMOTE_BLOCK)
        next_placed = set()

        for bx, by in relative_blocks:
            wx = MC_REMOTE_ORIGIN_X + bx
            wy = MC_REMOTE_ORIGIN_Y + by
            wz = MC_REMOTE_ORIGIN_Z
            self.mc.setBlock(wx, wy, wz, block_name)
            next_placed.add((wx, wy, wz))

        self.placed = next_placed
        return len(self.placed)

    def close(self) -> None:
        if self.mc is None:
            return
        try:
            self.mc.close()
        except Exception:
            pass


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera open failed")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    writer = MCShapeWriter()
    writer.connect()

    trace_points: List[Tuple[float, float]] = []
    scale = 1.0
    last_thumb_y = None
    prev_open_hand = False
    last_rebuild_time = 0.0
    block_count = 0
    status_text = "Ready"

    cv2.namedWindow(WINDOW_NAME)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        open_hand = False
        fist = False
        thumb_up = False
        pinch = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            state = finger_states(hand)
            open_hand = is_open_hand(state)
            fist = is_fist(state)
            thumb_up = is_thumb_up(hand, state)
            pinch = is_pinch(hand)

            tip = hand.landmark[8]
            px = tip.x * WIDTH
            py = tip.y * HEIGHT

            if thumb_up:
                if last_thumb_y is not None:
                    dy = last_thumb_y - tip.y
                    scale = max(SCALE_MIN, min(SCALE_MAX, scale + dy * SCALE_SENSITIVITY))
                last_thumb_y = tip.y
            else:
                last_thumb_y = None

            if pinch and not thumb_up:
                p = (px, py)
                if not trace_points or dist(trace_points[-1], p) >= MIN_TRACE_DISTANCE_PX:
                    trace_points.append(p)

            if fist:
                trace_points.clear()
                block_count = 0
                writer.clear()
                status_text = "Cleared by fist"

            if open_hand and not prev_open_hand and len(trace_points) > 1:
                now = time.time()
                if now - last_rebuild_time > 0.35:
                    scaled_points = scale_points(trace_points, scale)
                    rel_blocks = to_relative_blocks(scaled_points)
                    block_count = writer.rebuild_shape(rel_blocks)
                    last_rebuild_time = now
                    if writer.active:
                        status_text = f"Rebuilt by open hand: {block_count} blocks"
                    else:
                        status_text = "MC disconnected"

        scaled_preview = scale_points(trace_points, scale)
        draw_polyline(frame, trace_points, (0, 190, 255), 2)
        draw_polyline(frame, scaled_preview, (0, 255, 0), 2)

        if scaled_preview:
            p = scaled_preview[-1]
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

        cv2.circle(frame, (WIDTH // 2, HEIGHT // 2), 3, (255, 255, 255), -1)
        put_help(frame, scale, len(trace_points), block_count, writer.active, status_text)

        cv2.imshow(WINDOW_NAME, frame)
        prev_open_hand = open_hand

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("b"):
            if len(trace_points) > 1:
                scaled_points = scale_points(trace_points, scale)
                rel_blocks = to_relative_blocks(scaled_points)
                block_count = writer.rebuild_shape(rel_blocks)
                if writer.active:
                    status_text = f"Rebuilt by key: {block_count} blocks"
                else:
                    status_text = "MC disconnected"
            else:
                status_text = "Not enough points"
        if key == ord("c"):
            writer.clear()
            block_count = 0
            status_text = "Cleared by key"

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    writer.close()


if __name__ == "__main__":
    main()
