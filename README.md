# 📷 基於 Intel RealSense 與 MediaPipe 的即時 3D 動作辨識系統 (Real-time 3D Gesture & Action Recognition)

本專案利用 **Intel RealSense 深度攝影機** 的深度感知能力，結合 **Google MediaPipe** 的高效率人體姿態與手部骨架追蹤技術（Landmark Detection），實作了一套即時的動作與手勢辨識系統。本專案可用於人機互動（HCI）、智慧空間導航、動作培訓或 VR 系統控制。

---

## 🛠️ 技術與硬體需求 (Tech Stack & Hardware)
- **硬體設備**: Intel RealSense 深度攝影機 (如 D435i / D415)
- **核心語言**: Python 3.9+
- **電腦視覺與 AI 庫**: 
  - `pyrealsense2` (Intel RealSense 官方 SDK)
  - `mediapipe` (骨架與關鍵點偵測模型)
  - `opencv-python` (影像預處理、繪圖與視窗視覺化)

---

## 🚀 核心功能與辨識動作

系統透過 RealSense 獲取高畫質 RGB 影像與深度資訊（Depth Map），並由 MediaPipe 擷取關鍵點座標，目前主要聚焦於以下**三大核心動作辨識**：
1. **[動作一名稱]**: (例如：揮手 / Wave - 說明觸發邏輯，如手部高於肩膀)
2. **[動作二名稱]**: (例如：點擊 / Click - 說明觸發邏輯)
3. **[動作三名稱]**: (例如：雙手平舉 / T-Pose - 說明觸發邏輯)

---

## 📁 建議的專案資料夾結構

為了保持專案整潔，並確保每次修改 Code 都能被 Git 完美追蹤，建議結構如下：

```text
realsense-mediapipe-gesture-recognition/
│
├── .gitignore              (排除本地測試影片、暫存檔與虛擬環境)
├── README.md               (本說明文件)
├── requirements.txt        (環境套件依賴清單)
│
├── src/                    (原始碼資料夾)
│   ├── config.py           (存放 RealSense 分辨率、MediaPipe 信心度等參數)
│   ├── utils.py            (存放畫骨架、計算角度等輔助函式)
│   └── main.py             (主程式：負責驅動相機、模型推論與動作判斷)
│
└── assets/                 (專案展示多媒體)
    └── demo.jpg            (可以把你的執行畫面截圖放在這，展示在 README 中)
