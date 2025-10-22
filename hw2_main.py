import argparse, os, json, pathlib, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def load_or_synthesize(csv_path: str, target: str, seed: int = 42):
    rng = np.random.RandomState(seed)
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in columns: {list(df.columns)[:10]}...")
        return df, True
    # synthesize: 12 features (10 informative), 2000 samples
    n, p = 2000, 12
    X = rng.normal(size=(n, p))
    true_w = rng.uniform(-3, 3, size=(p, 1))
    y = X @ true_w + rng.normal(0, 2.0, size=(n, 1))
    cols = [f"f{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.ravel()
    return df, False

def split_features_target(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def metrics(y_true, y_pred):
    return dict(
        r2 = float(r2_score(y_true, y_pred)),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae = float(mean_absolute_error(y_true, y_pred))
    )

def plot_true_pred(y_true, y_pred, out_path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    m = max(y_true.max(), y_pred.max()); mi = min(y_true.min(), y_pred.min())
    plt.plot([mi, m],[mi, m], ls="--")
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("True vs Predicted")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_residuals(y_true, y_pred, out_path):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0, ls="--")
    plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title("Residual Plot")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def ci_pi_intervals(X_test_std, y_test, out_path):
    X_design = sm.add_constant(X_test_std)
    model_sm = sm.OLS(y_test, X_design).fit()
    pred = model_sm.get_prediction(X_design)
    frame = pred.summary_frame(alpha=0.05)  # 95% CI/PI
    # Plot mean prediction with CI/PI ribbon vs index
    import numpy as np
    idx = np.arange(len(frame))
    plt.figure()
    plt.plot(idx, frame["mean"], label="Prediction (mean)")
    plt.fill_between(idx, frame["mean_ci_lower"], frame["mean_ci_upper"], alpha=0.3, label="95% CI")
    plt.fill_between(idx, frame["obs_ci_lower"], frame["obs_ci_upper"], alpha=0.2, label="95% PI")
    plt.legend(); plt.xlabel("Test sample index"); plt.ylabel("y")
    plt.title("Prediction with 95% CI / PI")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return frame

def main(args):
    out_dir = pathlib.Path("outputs"); out_dir.mkdir(exist_ok=True)
    df, is_real = load_or_synthesize(args.data, args.target, args.seed)
    X, y, num_cols, cat_cols = split_features_target(df, args.target if is_real else "target")

    # transformers
    numeric_tf = Pipeline([("scaler", StandardScaler())])
    categorical_tf = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    preproc = ColumnTransformer(
        transformers=[("num", numeric_tf, num_cols), ("cat", categorical_tf, cat_cols)],
        remainder="drop"
    )

    base_lr = LinearRegression()

    # feature selection
    if args.selector == "kbest":
        selector = SelectKBest(score_func=f_regression, k=args.k)
    elif args.selector == "rfe":
        selector = RFE(estimator=LinearRegression(), n_features_to_select=args.k)
    elif args.selector == "lasso":
        selector = LassoCV(cv=5, random_state=args.seed)
    else:
        selector = "passthrough"

    pipe = Pipeline([("preproc", preproc), ("selector", selector), ("model", base_lr)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    pipe.fit(X_train, y_train)
    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)

    m_tr = metrics(y_train, y_pred_tr)
    m_te = metrics(y_test, y_pred_te)

    # CV (optional)
    cv_scores = None
    if args.cv and args.cv > 1:
        from sklearn.model_selection import KFold, cross_val_score
        cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(pipe, X, y, scoring="r2", cv=cv).tolist()

    # Visualizations
    plot_true_pred(y_test, y_pred_te, out_dir/"true_vs_pred.png")
    plot_residuals(y_test, y_pred_te, out_dir/"residuals.png")

    # Intervals using statsmodels on standardized numeric features of test set
    if len(num_cols) > 0:
        scaler_only = StandardScaler().fit(X_train[num_cols])
        X_test_std = scaler_only.transform(X_test[num_cols])
        interval_df = ci_pi_intervals(X_test_std, y_test.to_numpy(), out_dir/"prediction_intervals.png")
        interval_csv = out_dir/"prediction_intervals.csv"; interval_df.to_csv(interval_csv, index=False)
    else:
        interval_csv = None

    result = {
        "data_used": "real" if is_real else "synthetic",
        "shape": {"n_samples": int(df.shape[0]), "n_features": int(df.shape[1]-1)},
        "selector": args.selector, "k": args.k,
        "metrics_train": m_tr, "metrics_test": m_te,
        "cv_r2_scores": cv_scores,
        "artifacts": {
            "true_vs_pred": str(out_dir/"true_vs_pred.png"),
            "residuals": str(out_dir/"residuals.png"),
            "prediction_intervals_plot": str(out_dir/"prediction_intervals.png") if interval_csv else None,
            "prediction_intervals_csv": str(out_dir/"prediction_intervals.csv") if interval_csv else None,
        }
    }
    (out_dir/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/winequality-red.csv")
    ap.add_argument("--target", type=str, default="quality")
    ap.add_argument("--selector", type=str, choices=["none","kbest","rfe","lasso"], default="kbest")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=0)
    args = ap.parse_args()
    if args.selector == "none":
        args.selector = "passthrough"
    main(args)
