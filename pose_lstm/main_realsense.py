import os
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import time
import threading
import pandas as pd
from datetime import datetime
from collections import deque, Counter
from tensorflow.keras.models import load_model
from config import *

RECORD_MODE = False

# ── 全域狀態 ──────────────────────────────────────────────
current_state = "STANDING"
pose_start_time = None
camId = 1
last_sent_pose = None
display_running = True

# ── 平滑快取 ──────────────────────────────────────────────
SMOOTHING_WINDOW = 8
knee_angle_buf      = deque(maxlen=SMOOTHING_WINDOW)
hip_depth_diff_buf  = deque(maxlen=SMOOTHING_WINDOW)
knee_diff_buf       = deque(maxlen=SMOOTHING_WINDOW)
foot_height_diff_buf= deque(maxlen=SMOOTHING_WINDOW)

# ── LSTM 快取 ─────────────────────────────────────────────
lstm_feature_window = deque(maxlen=N_TIME)
pose_history        = deque(maxlen=20)
ai_predicted_label  = "None"
ai_confidence       = 0.0          # ← 新增：儲存最高信心值

classes = []
model   = None
record_label  = "Unknown"
recorded_data = []

# ── MediaPipe 骨架連線定義（下半身 + 軀幹）────────────────
MP_CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,       mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,       mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,      mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,      mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,     mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]

# ── 顯示參數 ──────────────────────────────────────────────
LABEL_COLORS = {
    "STANDING":            (200, 200, 200),
    "squat":               (0,   200, 255),
    "knee_propping":       (0,   230, 100),
    "SL_forward_stepping": (255, 180,   0),
    "None":                (160, 160, 160),
}

# ==========================================================
# 🚀 初始化
# ==========================================================
print("⏳ 初始化 MediaPipe ...")
mp_pose   = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("⏳ 啟動 RealSense 管線 ...")
pipeline  = rs.pipeline()
config_rs = rs.config()
config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config_rs)
align = rs.align(rs.stream.color)

PROCESS_EVERY = 2
last_results  = None
frame_count   = 0

# 載入模型
if not RECORD_MODE:
    model_path = os.path.join(MODEL_DIR, 'best.h5')
    if os.path.exists(model_path):
        model   = load_model(model_path)
        files   = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        classes = sorted([f.split('.')[0] for f in files])
        print(f"🤖 模型載入成功！類別: {classes}")
    else:
        print("⚠️  找不到 best.h5，將以純規則模式執行")

# ==========================================================
# 工具函式
# ==========================================================
def async_lstm_inference(feature_snapshot):
    global ai_predicted_label, ai_confidence
    if model is None: return
    tensor = np.expand_dims(feature_snapshot, axis=0)
    result = model.predict(tensor, verbose=0)
    ai_confidence      = float(np.max(result[0]))
    ai_predicted_label = classes[np.argmax(result[0])]

def get_region_depth(depth_frame, cx, cy, radius=3):
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if cx < 0 or cx >= dw or cy < 0 or cy >= dh: return 0.0
    depths = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < dw and 0 <= ny < dh:
                d = depth_frame.get_distance(nx, ny)
                if d > 0.01: depths.append(d)
    return float(np.median(depths)) if len(depths) >= 3 else 0.0

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians  = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle    = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ==========================================================
# 視覺化輔助函式
# ==========================================================
def draw_skeleton(img, lm, w, h, color=(0, 255, 128)):
    """繪製骨架連線與關節點"""
    for (a, b) in MP_CONNECTIONS:
        pa = (int(lm[a.value].x * w), int(lm[a.value].y * h))
        pb = (int(lm[b.value].x * w), int(lm[b.value].y * h))
        if lm[a.value].visibility > 0.3 and lm[b.value].visibility > 0.3:
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    # 關節圓點
    key_joints = [
        mp_pose.PoseLandmark.LEFT_HIP,  mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    ]
    for jt in key_joints:
        p = (int(lm[jt.value].x * w), int(lm[jt.value].y * h))
        if lm[jt.value].visibility > 0.3:
            cv2.circle(img, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, 5, color, 1, cv2.LINE_AA)


def draw_angle_arc(img, vertex, angle_deg, radius=28, color=(255, 200, 0)):
    """在膝蓋位置畫一個角度弧線"""
    start_angle = -int(angle_deg / 2)
    end_angle   =  int(angle_deg / 2)
    cv2.ellipse(img, vertex, (radius, radius), -90,
                start_angle, end_angle, color, 2, cv2.LINE_AA)


def draw_hud(img, knee_angle, hip_depth_diff, knee_diff,
             foot_diff, detected_pose, confidence, duration):
    """繪製半透明 HUD 面板"""
    h, w = img.shape[:2]

    # ── 上方大標籤：辨識結果 ────────────────────────────
    label_color = LABEL_COLORS.get(detected_pose, (200, 200, 200))
    label_text  = detected_pose if detected_pose != "None" else "偵測中..."

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, label_text, (14, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, label_color, 2, cv2.LINE_AA)

    # 信心條（右上）
    bar_x, bar_y, bar_w, bar_h = w - 170, 10, 150, 18
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * confidence)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), label_color, -1)
    cv2.putText(img, f"{confidence*100:.0f}%", (bar_x + bar_w + 4, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ── 下方數值面板 ─────────────────────────────────────
    panel_h = 110
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, img, 0.35, 0, img)

    def row(label, val_str, y, hi_color=None):
        cv2.putText(img, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        color = hi_color if hi_color else (230, 230, 230)
        cv2.putText(img, val_str, (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    base_y = h - panel_h + 22
    row("膝蓋角度 Knee Angle",   f"{knee_angle:.1f} deg", base_y,
        (0,200,80) if knee_angle < 150 else None)
    row("髖深差 Hip Depth Diff", f"{hip_depth_diff*100:.1f} cm",  base_y + 24,
        (0,200,255) if hip_depth_diff > 0.05 else None)
    row("左右膝差 Knee Diff",    f"{knee_diff:.1f} deg",  base_y + 48,
        (255,180,0) if knee_diff > 15 else None)
    row("腳高差 Foot Height Diff",f"{foot_diff:.0f} px",   base_y + 72,
        (255,100,100) if foot_diff > 40 else None)

    # 持續時間
    if duration > 0:
        cv2.putText(img, f"持續 {duration:.1f}s", (w - 130, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # ── 按鍵提示 ─────────────────────────────────────────
    cv2.putText(img, "Q: 離開", (w - 85, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)


# ==========================================================
# 🚀 主循環
# ==========================================================
cv2.namedWindow("ITRI Pose Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ITRI Pose Monitor", 800, 600)

try:
    while display_running:
        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame: continue

        frame_count += 1
        color_image  = np.asanyarray(color_frame.get_data())
        h, w, _      = color_image.shape
        rgb_image    = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY == 0:
            results      = pose_model.process(rgb_image)
            last_results = results
        else:
            results = last_results

        detected_pose  = "STANDING"
        knee_angle = hip_depth_diff = knee_diff = foot_height_diff = duration = 0.0

        if results is not None and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            required = [
                mp_pose.PoseLandmark.RIGHT_HIP.value,  mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value,mp_pose.PoseLandmark.LEFT_ANKLE.value,
            ]

            # ── 繪製骨架（永遠繪製，不論角度是否計算成功）──
            draw_skeleton(color_image, lm, w, h)

            if min(lm[i].visibility for i in required) >= 0.4:
                def lm_px(idx): return int(lm[idx].x * w), int(lm[idx].y * h)

                r_hip,  l_hip  = lm_px(mp_pose.PoseLandmark.RIGHT_HIP.value),  lm_px(mp_pose.PoseLandmark.LEFT_HIP.value)
                r_knee, l_knee = lm_px(mp_pose.PoseLandmark.RIGHT_KNEE.value),  lm_px(mp_pose.PoseLandmark.LEFT_KNEE.value)
                r_ank,  l_ank  = lm_px(mp_pose.PoseLandmark.RIGHT_ANKLE.value), lm_px(mp_pose.PoseLandmark.LEFT_ANKLE.value)

                foot_height_diff_raw = abs(r_ank[1] - l_ank[1])
                r_hip_d = get_region_depth(depth_frame, *r_hip)
                l_hip_d = get_region_depth(depth_frame, *l_hip)

                if r_hip_d > 0.0 and l_hip_d > 0.0:
                    r_ka = calculate_angle(r_hip, r_knee, r_ank)
                    l_ka = calculate_angle(l_hip, l_knee, l_ank)

                    knee_angle_raw      = (r_ka + l_ka) / 2
                    hip_depth_diff_raw  = abs(r_hip_d - l_hip_d)
                    knee_diff_raw       = abs(r_ka - l_ka)

                    knee_angle_buf.append(knee_angle_raw)
                    hip_depth_diff_buf.append(hip_depth_diff_raw)
                    knee_diff_buf.append(knee_diff_raw)
                    foot_height_diff_buf.append(foot_height_diff_raw)

                    knee_angle       = np.mean(knee_angle_buf)
                    hip_depth_diff   = np.mean(hip_depth_diff_buf)
                    knee_diff        = np.mean(knee_diff_buf)
                    foot_height_diff = np.mean(foot_height_diff_buf)
                    knee_angle_std   = float(np.std(list(knee_angle_buf))) if len(knee_angle_buf) >= 4 else 0.0

                    # ── 膝蓋角度弧線 ──────────────────────
                    draw_angle_arc(color_image, r_knee, r_ka, color=(255, 220, 0))
                    draw_angle_arc(color_image, l_knee, l_ka, color=(255, 220, 0))

                    # ── 角度文字標注 ──────────────────────
                    cv2.putText(color_image, f"{r_ka:.0f}",
                                (r_knee[0] + 30, r_knee[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)
                    cv2.putText(color_image, f"{l_ka:.0f}",
                                (l_knee[0] - 55, l_knee[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)

                    current_features = [
                        knee_angle, hip_depth_diff, knee_diff, foot_height_diff, knee_angle_std,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                        lm[mp_pose.PoseLandmark.NOSE.value].y
                    ]

                    if not RECORD_MODE:
                        lstm_feature_window.append(current_features)
                        if len(lstm_feature_window) == N_TIME:
                            feature_snapshot = np.array(list(lstm_feature_window))
                            t = threading.Thread(target=async_lstm_inference, args=(feature_snapshot,))
                            t.daemon = True
                            t.start()

                        pose_history.append(ai_predicted_label if ai_predicted_label != "None" else "STANDING")
                        detected_pose = Counter(pose_history).most_common(1)[0][0]

                        if current_state != detected_pose:
                            current_state   = detected_pose
                            pose_start_time = time.time()
                            print(f"\n📡 [狀態變更] → {current_state}  (膝蓋角度: {knee_angle:.1f}°)")

                        if pose_start_time:
                            duration = time.time() - pose_start_time

        # ── 繪製 HUD ──────────────────────────────────────
        draw_hud(color_image, knee_angle, hip_depth_diff, knee_diff,
                 foot_height_diff, detected_pose, ai_confidence, duration)

        cv2.imshow("ITRI Pose Monitor", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q 或 ESC 離開
            print("\n👋 使用者按下 Q，程式結束。")
            break

except KeyboardInterrupt:
    print("\n👋 Ctrl+C 終止。")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()