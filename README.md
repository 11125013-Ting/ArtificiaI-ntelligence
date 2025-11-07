# 人工智慧期中作業：Google Colab 影片目標追蹤

## 組員
- 11125013 郭慧庭
- 11125032 林欣儀
- 11125036 夏振凱

---

## 一、前言與程式目的

本次作業的目標是使用 **Google Colab** 執行「影片目標追蹤」。  
我們原先依照題目教學示範流程操作，但因環境差異，原始方法在 **模型推論階段無法正常執行**。  
因此本組後續改以其他能在 Colab 正常執行的追蹤方法進行測試與比較。

- **題目教學來源：**  
  https://blog.csdn.net/qq_30347421/article/details/104534297

- **測試影片來源（免版權可下載使用）：**  
  https://pixabay.com/videos/search/people%20walking/?utm_source=chatgpt.com

---

## 二、題目原始方法
> 此章節預留位置，用於補上：  
> - 原始流程操作截圖  
> - 中斷 / 錯誤狀況說明  
> - 與可執行方法比較  
（後續完成後再補）

---

## 三、可在 Colab 成功執行的方法

下列三種方法皆可在 Google Colab 正常執行。  
執行前建議先準備測試影片，並掛載 Google Drive 或直接上傳影片。

### 共同前置作業

**掛載 Google Drive：**
```python
from google.colab import drive
drive.mount('/content/drive')
```
**或直接上傳影片：**
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 方法 1：自動找人 + CSRT 追蹤
**Notebook：**
https://colab.research.google.com/drive/1myJMZpZqiKZTzWcI_MTvekDEtF_hy_na

**概念說明：**
- 使用 HOG 在影片「第一幀」偵測人物位置
- 擷取偵測框作為初始追蹤框
- 使用 CSRT 追蹤器逐幀追蹤
- 輸出完成追蹤的影片

**安裝環境**
```bash
!pip install -q opencv-contrib-python-headless==4.10.0.84 numpy==1.21.2
```

**輸出檔案**
![videos/auto_csrt_tracked.mp4](auto_csrt_tracked.mp4)


**執行成果**
- **輸出影片：** 
  ![videos/auto_csrt_tracked.mp4](auto_csrt_tracked.mp4)
- **成果截圖：**  
  ![]()

**常見錯誤與處理方式**
| 錯誤訊息 | 原因 | 解決方式 |
|---|---|---|
| `AttributeError: module 'cv2.legacy' has no attribute 'TrackerCSRT_create'` | 安裝版本不含 CSRT 追蹤模組 | 重新安裝：`opencv-contrib-python-headless==4.10.0.84` |

---

### 方法 2：骨架偵測追蹤（MediaPipe Pose）
**Notebook：**
https://colab.research.google.com/drive/1H91ZppZwKA_QGpaH-PmXRAr2GZKtwkSJ

**概念說明**
- 使用 MediaPipe Pose 偵測 33 個人體關鍵點
- 以線段連接形成骨架，觀測動作姿勢
- 適合 舞蹈、運動、動作分析

**安裝環境**
```bash
!pip install -q mediapipe==0.10.14 opencv-python-headless==4.10.0.84
```

**輸出檔案**
![videos/pose_tracked.mp4](pose_tracked.mp4)


**執行成果**
- **輸出影片：** 
  ![videos/pose_tracked.mp4](pose_tracked.mp4)
- **骨架標示：**  
  ![]()

**常見狀況與處理方式**
| 問題 | 原因 | 解決方式 |
|---|---|---|
| 骨架點會忽隱忽現、線段抖動 | 偵測信心值偏低 | 將 `min_detection_confidence` 與 `min_tracking_confidence` 調至 `0.6~0.7` |
| 播放卡頓 / FPS 不穩定 | 寫檔時 FPS 未設定 | 以 `cv2.VideoWriter` 固定 FPS 或沿用原影片 FPS |

---

### 方法 3：YOLO 偵測 + 多人 CSRT 追蹤
**Notebook：**
https://colab.research.google.com/drive/1Q5uEcF9hB27QALWkMBR2JAYDF9GiEdd3


**概念說明**
- 使用 YOLOv8 偵測畫面中所有人物
- 對每位人物建立獨立追蹤器
- 每隔數幀重新偵測，避免追蹤偏移
- 可同時追蹤多人

**安裝環境**
```bash
!pip install -q ultralytics==8.2.103 opencv-contrib-python-headless==4.10.0.84
```

**輸出檔案**
![videos/test-2_tracked.mp4](test-2_tracked.mp4)

**執行成果**
- **輸出影片：**
  ![videos/test-2_tracked.mp4](test-2_tracked.mp4)
- **追蹤截圖：**  
  ![]()

**常見狀況與處理方式**
| 問題 | 原因 | 解決方式 |
|---|---|---|
| 追蹤速度慢 / 延遲 | YOLO 模型較大 & 偵測過頻 | 改用 `yolov8n.pt` 或將 `DETECT_INTERVAL` 增至 `10~15` |
| 偵測不到人或框跳動 | 影片清晰度不足或 threshold 太低 | 提高 YOLO confidence threshold，並確保影片清晰度 |

---

## 四、三種方法比較

| 方法 | 偵測方式 | 優點 | 缺點 | 適合情境 |
|---|---|---|---|---|
| 方法 1 | HOG + CSRT | 單人追蹤穩定、設定簡單 | 不支援多人 | **單人追蹤影片** |
| 方法 2 | MediaPipe Pose | 能呈現人體動作與姿勢 | 骨架可能抖動、對光影敏感 | **舞蹈 / 運動分析** |
| 方法 3 | YOLO + 多 CSRT | 可多人同時追蹤 | 計算量較大、速度較慢 | **多人群體畫面** |

---

## 五、執行截圖（本組實際紀錄）

| 描述 | 檔案 |
|---|---|
| 掛載 Google Drive 成功 | `assets/step2_mount.jpg` |
| 追蹤成果畫面 | `assets/step5_result.jpg` |

> 註：`step5_result.jpg` 對應方法 1 成果影片中的追蹤框畫面。

---

## 六、結論與使用建議

- **方法 1（HOG + CSRT）**  
  執行最穩定、步驟最少，適合初次實作與單人追蹤。

- **方法 2（MediaPipe Pose）**  
  可顯示人體關節與姿勢，適用需觀察動作的情境。

- **方法 3（YOLO + 多 CSRT）**  
  可多人追蹤，但運算較慢，適合複雜、多角色畫面。

**使用建議：**
| 需求 | 建議方法 |
|---|---|
| 想要成功率高、快完成 | 方法 1 |
| 想看人體動作 / 教學展示 | 方法 2 |
| 場景中有多人 | 方法 3 |

---
