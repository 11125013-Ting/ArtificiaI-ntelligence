# 人工智慧期中作業：Google Colab 影片目標追蹤

## 組員
- 11125013 郭慧庭
- 11125032 林欣儀
- 11125036 夏振凱

---

## 一、作業目的
本作業的目標是在 Google Colab 上執行影片目標追蹤。  
我們先依照題目提供的示範流程操作，但因環境差異導致模型推論階段無法正常執行，  
因此補充三種可在 Colab 正常執行的替代追蹤方法，並整理執行流程與結果。

題目教學來源：  
https://blog.csdn.net/qq_30347421/article/details/104534297

---

## 二、題目原始方法（無法於 Colab 完整執行）
Notebook：  
https://colab.research.google.com/drive/1W4ejb55Ll4tb3B0jU2w1ABNdjUYuo6yi#scrollTo=S8xKLIJKQ14C

本組依照題目流程在 Colab 中設定環境與執行程式，並依執行狀況調整了路徑及套件版本。  
前置流程可成功執行，但在模型推論階段因環境版本差異，最終無法輸出追蹤結果影片。

主要執行流程如下：

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/video_analyst/
pip install -r requirements.txt
python tools/test.py
```

> **結果說明：**  
> 程式可啟動並執行前半段流程，但在讀取模型與推論階段中斷，未能成功產生追蹤影片。

本組原始實作紀錄：
https://drive.google.com/xxxxx

---

## 三、可在 Colab 成功執行的方法

### 方法 1：自動找人 + CSRT 追蹤
Notebook：  
https://colab.research.google.com/drive/1UuaKV3uMsmgFQVUwjKzzMdos_z6g7PwS

**概念**：先用 HOG 偵測第一幀的人 → 取得初始邊界框 → 用 CSRT 追蹤整段影片  
**成果檔案**  
- 影片：[`videos/auto_csrt_tracked.mp4`](videos/auto_csrt_tracked.mp4)  
- 截圖：![`assets/step5_result.jpg`](assets/step5_result.jpg)

---

### 方法 2：骨架偵測追蹤
Notebook：  
https://colab.research.google.com/drive/17CKV5CozvxaJSQ1eyOoN0NbEIr0gJrpT

**概念**：不框人，使用 MediaPipe 偵測身體關鍵點並繪製骨架  
**輸出影片**：`pose_tracked.mp4`

---

### 方法 3：YOLO 偵測 + 多人 CSRT 追蹤
Notebook：  
https://colab.research.google.com/drive/1xLaj-yQcLALdZA7mFPYs2ryNsWB-gkPR

**概念**：每隔數幀重新偵測人物 → 為每個人建立追蹤器 → 多人可同時追蹤  
**輸出影片**：依程式設定，例如：`test-2_tracked.mp4`

---

## 四、三種方法比較

| 方法 | 特點 | 適合情況 | 追蹤對象數 |
|---|---|---|---|
| 方法 1 | 先偵測一次後持續追蹤 | 單人、畫面穩定 | 一人 |
| 方法 2 | 偵測骨架，不使用邊界框 | 活動動作分析 | 一人或多人 |
| 方法 3 | 重新偵測 + 多追蹤器 | 多人同時出現在畫面中 | 多人 |

---

## 五、執行結果截圖（稍後補）

| 描述 | 檔名（建議放在 `assets/` 資料夾） |
|---|---|
| Drive 掛載成功 | `step2_mount.jpg` |
| 切換資料夾成功 | `step3_cd.jpg` |
| 套件安裝畫面 | `step4_pip.jpg` |
| 追蹤或骨架標示成功畫面 | `step5_result.jpg` |

---
