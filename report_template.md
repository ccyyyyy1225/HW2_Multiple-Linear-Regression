# HW2 報告 — Multiple Linear Regression（CRISP-DM 流程）

## 🧩 資料集來源
- **資料集名稱**：Wine Quality Dataset（UCI Machine Learning Repository）
- **Kaggle 連結**：[https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **資料摘要**：  
  此資料集包含葡萄酒的 11 項理化特徵（如酸度、糖分、酒精濃度等）與品質評分（0–10 分）。  
  本作業以「紅酒資料集 (winequality-red.csv)」作為分析對象，  
  總樣本數約 1600 筆，目標欄位為 `quality`。

---

## 1️⃣ Business Understanding
紅酒品質預測是回歸問題中的典型案例，目標在於建立一個能根據理化特徵（例如酸度、糖分、酒精濃度等）  
預測品質評分的模型。此問題可應用於：
- 酒莊品質分級自動化  
- 生產線品質控管  
- 原料調配建議系統  

成功準則包括：
- 預測準確度（R² 越高越好）  
- 誤差（RMSE、MAE 越低越好）  
- 模型可解釋性（係數方向合理）  

---

## 2️⃣ Data Understanding
1. **基本資料**
   - 特徵數：11 個  
   - 目標：`quality`（整數 0–10）  
   - 資料筆數：約 1599 筆  

2. **特徵摘要**
   | 特徵名稱 | 含義 | 單位 |
   |-----------|------|------|
   | fixed acidity | 固定酸度 | g(tartaric acid)/dm³ |
   | volatile acidity | 揮發性酸度 | g(acetic acid)/dm³ |
   | citric acid | 檸檬酸 | g/dm³ |
   | residual sugar | 殘糖 | g/dm³ |
   | chlorides | 氯化物 | g(sodium chloride)/dm³ |
   | free sulfur dioxide | 游離二氧化硫 | mg/dm³ |
   | total sulfur dioxide | 總二氧化硫 | mg/dm³ |
   | density | 密度 | g/cm³ |
   | pH | 酸鹼值 | – |
   | sulphates | 硫酸鹽 | g/potassium sulphate/dm³ |
   | alcohol | 酒精濃度 | % |

3. **初步觀察**
   - 酒精濃度與品質呈正相關。  
   - 揮發性酸度與品質呈負相關。  
   - 相關係數熱圖顯示前述兩者影響最明顯。  

---

## 3️⃣ Data Preparation
- **缺失值處理**：無遺失值。  
- **特徵工程**：
  - 將特徵標準化（StandardScaler）。  
  - 目標變數維持原始尺度。  
- **資料分割**：
  - 訓練集：75%，測試集：25%。  
  - 隨機種子：42。  
- **特徵選擇**：
  - 使用 SelectKBest（f_regression）選出 8 個最具線性關聯的特徵：
    - alcohol、volatile acidity、sulphates、citric acid、density、pH、fixed acidity、total sulfur dioxide。

---

## 4️⃣ Modeling
模型選擇：
- **模型類型**：Multiple Linear Regression  
- **特徵選擇策略**：SelectKBest (k=8)  
- **訓練流程**：
  1. 標準化 → 特徵選擇 → 線性回歸  
  2. 評估 Train/Test 表現差異  
  3. 顯示 True vs Predicted、Residual Plot、Prediction Intervals  

---

## 5️⃣ Evaluation
| 指標 | 訓練集 | 測試集 |
|------|--------|--------|
| R² | 0.3511 | 0.3651 |
| RMSE | 0.6557 | 0.6267 |
| MAE | 0.5033 | 0.5035 |

- 模型可解釋約 **35% 的資料變異**，在此資料噪聲水平下屬合理表現。  
- Train/Test R² 接近，顯示模型未過擬合。  
- RMSE 與 MAE 約 0.6，平均預測誤差小於 1 分品質分數。  
- 殘差圖呈隨機分佈，符合線性假設。  

**圖表說明：**
1. `true_vs_pred.png`：顯示實際值與預測值之間的線性關係。  
2. `residuals.png`：殘差分佈呈隨機，無明顯模式。  
3. `prediction_intervals.png`：以 `statsmodels` 估算 95% 信賴區間（CI）與預測區間（PI）。  
   - CI 代表平均預測值的不確定性。  
   - PI 代表新樣本落點範圍（通常更寬）。

> **結論**：模型表現穩定，具有解釋性與可預測性，適合用於基線回歸任務。若要提升準確率，可考慮非線性模型（如 Random Forest 或 Gradient Boosting）。

---

## 6️⃣ Deployment
- 以 **Streamlit** 製作互動式網頁應用。  
- 使用者可上傳任意 CSV，選擇特徵選擇法（KBest/RFE/Lasso），自動生成評估結果與預測區間。  
- 執行方式：
  ```bash
  streamlit run app/app.py
  ```
- 互動功能：
  - 上傳資料集
  - 調整參數（k、test_size、seed）
  - 顯示指標、圖表與預測帶
  - 下載 JSON / CSV 結果檔

---

## 🤖 GPT 輔助內容
本報告之專案結構、主程式、Streamlit 介面與 CRISP-DM 架構均由 ChatGPT（GPT-5）提供協助生成。  
AI 輔助內容包含：
- hw2_main.py 與 app.py 之程式模板  
- report_template.md 撰寫指引  
- 評估結果文字化分析（R²、RMSE、MAE 解釋）  
- Streamlit 部署建議與說明  

---

## 📚 NotebookLM 研究摘要
> 我使用 NotebookLM 搜尋與多元線性回歸特徵選擇相關的教學資料。  
> 多數研究指出，SelectKBest 適用於前期特徵過濾，可快速剔除無關變數；  
> RFE 透過遞迴式模型權重排名，對多重共線性處理較佳；  
> 而 Lasso 透過 L1 正則化自動產生稀疏解，有助於特徵壓縮與解釋性。  
> 在 Wine Quality 資料集中，由於特徵數僅 11 個，三種方法的效果接近，但 SelectKBest 計算速度快、結果穩定，適合作為基線方法。  
> 此外，多篇教學指出線性回歸假設包括：線性關係、獨立誤差、同方差性與殘差常態分佈，  
> 若違反假設則可考慮多項式擴充或樹模型以改善擬合度。

---

## 📑 參考資料
- Scikit-learn 官方文件：[https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Statsmodels 文件：[https://www.statsmodels.org/](https://www.statsmodels.org/)
- Kaggle: Red Wine Quality Dataset
- UCI Machine Learning Repository: Wine Quality Dataset
- ChatGPT（GPT-5）輔助產生與修訂內容
