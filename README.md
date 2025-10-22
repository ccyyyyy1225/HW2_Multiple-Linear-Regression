# HW2 — Multiple Linear Regression (CRISP-DM) with Feature Selection & Intervals

**交付包**：主程式（`.py` + `.ipynb`）、Streamlit 網頁、CRISP-DM 報告樣板、結果輸出（圖與 JSON）。  
**資料集**：建議使用 Kaggle「Wine Quality」(11 特徵) — 下載 `winequality-red.csv` 或 `winequality-white.csv` 放入 `data/`。

## 推薦資料集（10–20 特徵）
- **Wine Quality**（11 features, regression on `quality`）  
  Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

> 若尚未下載，程式會以 *合成資料* 自動跑通，確保可執行。

## 專案結構
```
4111056026_hw2/
├── app/
│   └── app.py                  # Streamlit 部署版（可視化 + 特徵選擇 + 評估）
├── data/                       # 放 CSV 資料（winequality-*.csv）
├── hw2_main.py                 # 主程式（.py）— 符合作業規格
├── hw2_notebook.ipynb          # 主程式（.ipynb）— Colab 友善
├── report_template.md          # 報告模板（CRISP-DM、GPT/NotebookLM 區塊）
├── requirements.txt
└── outputs/                    # 產出圖與結果（自動生成）
```

## 快速開始（本地）
```bash
pip install -r requirements.txt
python hw2_main.py --data data/winequality-red.csv --target quality --k 8
```

## Streamlit（部署/互動）
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 主要功能
- **CRISP-DM**：程式內與報告模板完整對應 6 步驟
- **特徵選擇**：SelectKBest(f_regression)、RFE、Lasso（可選）
- **評估**：R²、RMSE、MAE、KFold CV（可選）
- **區間**：以 `statsmodels` 計算信賴區間（mean CI）與預測區間（PI）
- **視覺化**：實際 vs. 預測、殘差圖、含區間的預測帶
- **可重現**：`--seed` 控制隨機性；所有超參數可調

## 注意
- 報告需自行加入 **GPT 對話 PDF（pdfCrowd 匯出）** 與 **NotebookLM 摘要**（模板中已有占位）。
- 若使用其他資料集，請更新 `--data` 路徑與 `--target` 名稱，並在報告註明來源連結。
