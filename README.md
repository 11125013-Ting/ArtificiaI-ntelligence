# 人工智慧期中作業：Google Colab 執行影片目標追蹤

## 組員
- 11125013 郭慧庭
- 11125032 林欣儀
- 11125036 夏振凱

---

## 一、執行目的
本次作業目標為：在 Google Colab 上執行影片目標追蹤程式，使電腦能在影片中辨識並持續追蹤同一物體。  
我們依照老師提供教學網站的流程進行嘗試，並整理出可重複執行的操作步驟。

教學來源：  
https://blog.csdn.net/qq_30347421/article/details/104534297

---

## 二、執行環境
| 項目 | 說明 |
|-----|-----|
| 平台 | Google Colab |
| 影像處理套件 | OpenCV |
| 檔案存放 | Google Drive |

---

## 三、操作流程

### Step 1. 開啟 Colab
開啟我們小組使用的 Notebook：  
https://colab.research.google.com/drive/1oq9gmI7Gh2I7Pi1LgNtHYqgEalj6aP-R

### Step 2. 掛載 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
