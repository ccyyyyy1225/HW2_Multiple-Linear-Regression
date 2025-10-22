import io, os, json, pathlib, numpy as np, pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(page_title="HW2 — Multiple Linear Regression", layout="wide")
st.title("HW2｜Multiple Linear Regression — CRISP-DM + Feature Selection + Intervals")

with st.expander("CRISP-DM 摘要", expanded=False):
    st.markdown("""
**1. Business Understanding：** 給定 10–20 特徵之資料集，建立線性回歸模型以預測連續目標。  
**2. Data Understanding：** 觀察統計、缺失、分佈與相關性。  
**3. Data Preparation：** 缺失處理、編碼、標準化、切分與特徵選擇。  
**4. Modeling：** Multiple Linear Regression +（RFE/Lasso/SelectKBest）。  
**5. Evaluation：** R²、RMSE、MAE、視覺化、**預測區間**。  
**6. Deployment：** 本頁即為可部署互動介面。
""")

st.sidebar.header("資料與參數")
uploaded = st.sidebar.file_uploader("上傳 CSV（如 winequality-red.csv）", type=["csv"])
target = st.sidebar.text_input("目標欄位名 (target)", value="quality")
test_size = st.sidebar.slider("測試比例", 0.1, 0.5, 0.2, 0.05)
selector = st.sidebar.selectbox("特徵選擇", ["kbest","rfe","lasso","none"], index=0)
k = st.sidebar.number_input("k (kbest/rfe 保留特徵數)", value=8, min_value=1, step=1)
seed = st.sidebar.number_input("random seed", value=42, step=1)

def load_df(file) -> pd.DataFrame:
    if file is not None:
        return pd.read_csv(file)
    else:
        st.info("未上傳文件，使用合成資料以示範。")
        rng = np.random.RandomState(seed)
        n, p = 800, 12
        X = rng.normal(size=(n, p))
        beta = rng.uniform(-2, 2, size=(p, 1))
        y = X @ beta + rng.normal(0, 1.5, size=(n, 1))
        cols = [f"f{i}" for i in range(p)]
        df = pd.DataFrame(X, columns=cols); df["target"] = y.ravel()
        return df

df = load_df(uploaded)

st.subheader("Data Understanding")
st.write(df.head())
st.write(df.describe())

if target not in df.columns:
    st.warning(f"找不到目標欄位 {target}，請確認 CSV 或更改輸入。")
    st.stop()

X = df.drop(columns=[target])
y = df[target]
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# correlation heatmap for numeric
if len(num_cols) > 1:
    import seaborn as sns
    fig = plt.figure(figsize=(5,4))
    sns.heatmap(df[num_cols+[target]].corr(), annot=False)
    st.pyplot(fig)

numeric_tf = Pipeline([("scaler", StandardScaler())])
categorical_tf = Pipeline([("ohe", OneHotEncoder(handle_unknown='ignore'))])
preproc = ColumnTransformer([("num", numeric_tf, num_cols), ("cat", categorical_tf, cat_cols)])

base_lr = LinearRegression()

if selector == "kbest":
    feat_sel = SelectKBest(score_func=f_regression, k=k)
elif selector == "rfe":
    feat_sel = RFE(estimator=LinearRegression(), n_features_to_select=k)
elif selector == "lasso":
    feat_sel = LassoCV(cv=5, random_state=seed)
else:
    feat_sel = "passthrough"

pipe = Pipeline([("preproc", preproc), ("selector", feat_sel), ("model", base_lr)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
pipe.fit(X_train, y_train)
y_pred_tr = pipe.predict(X_train); y_pred_te = pipe.predict(X_test)

def metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return dict(
        r2 = float(r2_score(y_true, y_pred)),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae = float(mean_absolute_error(y_true, y_pred))
    )
m_tr = metrics(y_train, y_pred_tr)
m_te = metrics(y_test, y_pred_te)

c1, c2 = st.columns(2)
with c1:
    st.metric("Train R²", f"{m_tr['r2']:.4f}")
    st.metric("Train RMSE", f"{m_tr['rmse']:.4f}")
    st.metric("Train MAE", f"{m_tr['mae']:.4f}")
with c2:
    st.metric("Test R²", f"{m_te['r2']:.4f}")
    st.metric("Test RMSE", f"{m_te['rmse']:.4f}")
    st.metric("Test MAE", f"{m_te['mae']:.4f}")

# true vs pred
fig1 = plt.figure()
plt.scatter(y_test, y_pred_te, alpha=0.6); 
mi, ma = min(y_test.min(), y_pred_te.min()), max(y_test.max(), y_pred_te.max())
plt.plot([mi, ma], [mi, ma], ls='--')
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("True vs Predicted")
st.pyplot(fig1)

# residuals
fig2 = plt.figure()
plt.scatter(y_pred_te, (y_test - y_pred_te), alpha=0.6)
plt.axhline(0, ls="--"); plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title("Residuals")
st.pyplot(fig2)

# Intervals with statsmodels (numeric standardized design)
if len(num_cols) > 0:
    scaler = StandardScaler().fit(X_train[num_cols])
    X_test_std = scaler.transform(X_test[num_cols])
    X_design = sm.add_constant(X_test_std)
    model_sm = sm.OLS(y_test, X_design).fit()
    pred = model_sm.get_prediction(X_design)
    frame = pred.summary_frame(alpha=0.05)
    idx = np.arange(len(frame))
    fig3 = plt.figure()
    plt.plot(idx, frame["mean"], label="Prediction (mean)")
    plt.fill_between(idx, frame["mean_ci_lower"], frame["mean_ci_upper"], alpha=0.3, label="95% CI")
    plt.fill_between(idx, frame["obs_ci_lower"], frame["obs_ci_upper"], alpha=0.2, label="95% PI")
    plt.legend(); plt.xlabel("Test sample index"); plt.ylabel("y"); plt.title("Prediction with 95% CI/PI")
    st.pyplot(fig3)
    st.download_button("下載預測區間 (CSV)", data=frame.to_csv(index=False), file_name="prediction_intervals.csv", mime="text/csv")

# Downloads
res = dict(selector=selector, k=k, seed=seed, test_size=test_size, metrics_train=m_tr, metrics_test=m_te)
st.download_button("下載結果 JSON", data=json.dumps(res, indent=2), file_name="results.json", mime="application/json")
