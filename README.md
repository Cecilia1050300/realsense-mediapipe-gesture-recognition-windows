# RealSense + MediaPipe 房務/清潔作業行為即時姿態辨識系統

本專案利用 **Intel RealSense 深度相機** 與 **Google MediaPipe Pose** 骨架追蹤技術，針對清潔與房務人員常見的作業姿態進行即時（Real-time）分類與監測。系統透過平滑緩衝區機制降低環境雜訊，並在動作維持達指定時間後，自動透過 **MQTT 協定** 發送 JSON 格式的識別事件，可用於人體工學職業安全評估或自動化作業進度追蹤。

---

## 🎯 辨識行為與幾何特徵定義

系統目前支援以下四種核心狀態的動態與靜態辨識：

| 狀態名稱 (`STATE`) | 實際清潔/房務動作 | 核心幾何演算法原理 |
| :--- | :--- | :--- |
| **`STANDING`** | 正常直立站立 | 雙腿平均膝蓋角度大於 $140^\circ$，且未觸發動態跨步。 |
| **`SL_forward_stepping`** | 吸塵器作業（跨步） | **動態優先權最高**。監測到雙膝平均角度大於 $115^\circ$，且在滑動時間視窗內膝蓋角度的**標準差（波動度） $> 2.5$**。 |
| **`squat`** | 收垃圾、撿拾物品（大彎腰下蹲） | 雙腿平均膝蓋角度 $< 140^\circ$（處於下蹲狀態），且透過肩膀與髖部中心點計算出之**軀幹傾斜角度（Torso Tilt） $\ge 25^\circ$**。 |
| **`knee_propping`** | 鋪床作業（半蹲挺胸） | 雙腿平均膝蓋角度 $< 140^\circ$（處於下蹲狀態），但**軀幹傾斜角度（Torso Tilt） $< 25^\circ$**，保持上半身挺直直立。 |

---

## 🚀 核心功能與技術亮點

1. **雙執行緒架構 (Multi-threading)**
   * **影像擷取與運算 Thread**：負責即時讀取 RealSense 深度與彩色影像、執行 MediaPipe 骨架偵測、計算 3D/2D 幾何角度。
   * **UI 渲染 Thread**：獨立負責 OpenCV 視窗繪製（更新率約 15 FPS），確保串流畫面順暢不卡頓。
2. **時域平滑濾波 (Temporal Smoothing Filter)**
   * 使用 `collections.deque` 維持 $8$ 幀（Frame）的滑動視窗，對膝蓋角度、左右膝差、軀幹傾斜角度進行**中位數與平均數平滑化**，徹底消除骨架抖動（Jittering）。
3. **穩定狀態投票機制 (State Voting Mechanism)**
   * 內建容量為 $20$ 幀的狀態歷史快取，透過 `Counter().most_common(1)` 進行多數決投票，避免單幀誤判導致狀態頻繁閃爍切換。
4. **標準化幾何計算 (Normalized Coordinates)**
   * 軀幹垂直角直接採用 MediaPipe 的 $0.0 \sim 1.0$ 原始比例空間進行 `arctan2` 運算，**擺脫相機解析度與長寬比（640x480）造成的幾何拉伸扭曲**，大幅提升現場跨人體辨識的穩定度。
5. **深度感測容錯防禦**
   * 整合區域深度中位數選取（Region Depth Median Filter），若短暫發生深度光點遺失或被手部遮擋，系統將自動啟動預測維持，防止 UI 數據卡死或錯誤切換。

---

## ⚙️ 環境配置與依賴項

本專案在 Linux 系統（相容 X11/Xcb 顯示伺服器）下開發，需準備以下環境：

### 1. 硬體需求
* Intel RealSense 深度相機（如 D435, D435i, D455 等）

### 2. 軟體套件安裝
請使用 `pip` 安裝以下必要的 Python 套件：
```bash
pip install pyrealsense2 mediapipe opencv-python numpy
