# RealSense + MediaPipe 3D 手勢與動作辨識系統 (Windows 完全體版本)

本專案專為 **Windows 10/11** 環境開發，結合 **Intel RealSense D435/D435i 深度相機** 與 **Google MediaPipe**，進行即時的 3D 骨架與關節點預測。並進一步透過 **LSTM（長短期記憶網路）** 深度學習模型，實現高準確度的即時動態手勢與動作辨識。

---

## 🚀 專案核心架構

```text
realsense-mediapipe-gesture-recognition-windows/
├── pose_lstm/
│   ├── main_realsense.py    # 核心主程式：RealSense 影像串流 + MediaPipe 骨架提取 + LSTM 預測
│   ├── train_lstm.py        # LSTM 模型訓練腳本
│   ├── action_model.h5      # 已訓練完成的 LSTM 動作辨識模型權重檔
│   └── data_collection.py   # 動作關鍵點數據收集工具
├── .gitignore                # 排除 venv 虛擬環境與本機暫存檔
└── README.md                 # 本說明文件
```

---

## 🛠️ Windows 環境相容性與套件清單

為了確保 RealSense 驅動與 MediaPipe/TensorFlow 在 Windows 上的相容性，本專案已在以下環境完整測試通過。請遵循此版號組態進行安裝：

- **作業系統**：Windows 10 / 11（64-bit）
- **Python 版本**：Python 3.10.x（建議 3.10.11，最為穩定）

### 核心相容套件版號

```text
protobuf==3.20.3
mediapipe==0.10.7
tensorflow==2.13.0
pyrealsense2==2.54.2.5684
opencv-python==4.8.1.78
```

---

## 💻 快速開始與執行步驟

### 1. 建立並啟用 Python 3.10 虛擬環境

打開 Windows PowerShell 或 Command Prompt，在專案根目錄下執行：

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. 安裝核心相容套件

```bash
pip install --upgrade pip
pip install protobuf==3.20.3
pip install mediapipe==0.10.7
pip install tensorflow==2.13.0
pip install pyrealsense2
pip install opencv-python
```

### 3. 啟動即時辨識系統

確保 Intel RealSense 相機已透過 USB 3.0 插槽連接至電腦，然後進入核心資料夾執行主程式：

```bash
cd pose_lstm
python main_realsense.py
```

---

## 📊 深度學習模型（LSTM）設計說明

本系統之動態動作辨識採用 **LSTM（Long Short-Term Memory）** 網路，能夠有效捕捉前後影格（Frames）之間的時間序列特徵（Temporal Features）：

- **特徵提取（Feature Extraction）**：透過 MediaPipe Hands/Pose 獲取 3D 關節點座標 $(x, y, z)$ 與可信度（Visibility）。
- **時序輸入（Sequence Input）**：以連續 30 影格（約 1 秒的動態過程）作為一個時間序列特徵向量。
- **模型預測（Inference）**：將特徵矩陣送入 `action_model.h5` 進行 Softmax 分類，即時在 OpenCV 視窗上輸出目前辨識出的動作名稱與信心指數。