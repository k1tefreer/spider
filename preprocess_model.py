# preprocess_model.py
import warnings
warnings.filterwarnings("ignore")

from config import DISTRICTS

# 自动构建编号 → 中文名映射
DISTRICT_NAME_MAP = {code: name for code, name in DISTRICTS}

from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== 解决中文乱码 ==================
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "raw_house_data.csv"


def load_data():
    df = pd.read_csv(CSV_FILE)
    print("[INFO] 原始数据 shape:", df.shape)
    print("[INFO] 列名:", df.columns.tolist())
    return df


# ================== 基础清洗 ==================
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "district", "title",
        "total_price_wan", "unit_price_yuan_m2", "area_m2",
        "floor_info", "build_year", "location", "raw_info",
    ]
    df = df[keep_cols].copy()

    # 目标值必须有：总价
    df = df.dropna(subset=["total_price_wan"])

    # 面积过滤（10 ~ 300㎡之间）
    df = df[(df["area_m2"] > 10) & (df["area_m2"] < 300)]

    # build_year 转数字
    df["build_year"] = pd.to_numeric(df["build_year"], errors="coerce")

    # 价格简单过滤
    df = df[(df["total_price_wan"] > 10) & (df["total_price_wan"] < 2000)]

    df = df.reset_index(drop=True)
    print("[INFO] 基础清洗后 shape:", df.shape)
    return df


# ================== 特征工程 ==================
def map_floor_level(text: str) -> float:
    if not isinstance(text, str):
        return np.nan
    if "低" in text:
        return 1
    if "中" in text:
        return 2
    if "高" in text:
        return 3
    if "顶" in text:
        return 4
    return np.nan


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    CURRENT_YEAR = 2024
    df["house_age"] = CURRENT_YEAR - df["build_year"]

    df["floor_level_num"] = df["floor_info"].map(map_floor_level)

    if "unit_price_yuan_m2" in df.columns:
        df["district_avg_unit_price"] = (
            df.groupby("district")["unit_price_yuan_m2"]
              .transform(lambda x: x.fillna(x.median()))
        )
    else:
        df["district_avg_unit_price"] = np.nan

    print("[INFO] 特征工程后示例:")
    print(df[[
        "total_price_wan", "area_m2", "house_age",
        "floor_level_num", "district_avg_unit_price", "district"
    ]].head())

    return df


# ================== 划分训练/测试 ==================
def prepare_train_test(df: pd.DataFrame):
    num_cols = [
        "area_m2",
        "house_age",
        "district_avg_unit_price",
        "floor_level_num",
    ]
    cat_cols = ["district"]

    X = df[num_cols + cat_cols].copy()
    y = df["total_price_wan"].copy()

    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("[INFO] 训练集大小:", X_train.shape, "测试集大小:", X_test.shape)
    return X_train, X_test, y_train, y_test, num_cols, cat_cols


# ================== 预处理 Pipeline（含缺失值填补） ==================
def build_preprocess_pipeline(num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor


# ================== 训练 & 评估 ==================
def evaluate_model(model, X_train, X_test, y_train, y_test, name="model"):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    def _metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    tr_mae, tr_rmse, tr_r2 = _metrics(y_train, y_pred_train)
    te_mae, te_rmse, te_r2 = _metrics(y_test, y_pred_test)

    print(f"\n==== {name} ====")
    print(f"Train MAE:  {tr_mae:.3f}  RMSE: {tr_rmse:.3f}  R2: {tr_r2:.4f}")
    print(f"Test  MAE:  {te_mae:.3f}  RMSE: {te_rmse:.3f}  R2: {te_r2:.4f}")

    return y_pred_test


# ================== 各种可视化 ==================
def plot_price_distribution(df: pd.DataFrame):
    """总价分布直方图"""
    plt.figure(figsize=(8, 5))
    df["total_price_wan"].hist(bins=30)
    plt.xlabel("总价（万）")
    plt.ylabel("房源数量")
    plt.title("二手房总价分布")
    plt.tight_layout()
    plt.show()


def plot_unit_price_distribution(df: pd.DataFrame):
    """单价分布直方图"""
    if "unit_price_yuan_m2" not in df.columns:
        print("[WARN] 无 unit_price_yuan_m2 列，跳过单价分布图")
        return
    plt.figure(figsize=(8, 5))
    df["unit_price_yuan_m2"].dropna().hist(bins=30)
    plt.xlabel("单价（元/㎡）")
    plt.ylabel("房源数量")
    plt.title("二手房单价分布")
    plt.tight_layout()
    plt.show()


def plot_unit_price_by_district(df):
    df["district_name"] = df["district"].astype(str).map(DISTRICT_NAME_MAP)

    plt.figure(figsize=(10, 6))
    df.boxplot(column="unit_price_yuan_m2", by="district_name", rot=45)
    plt.ylabel("单价（元/㎡）")
    plt.title("不同区域二手房单价分布")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()



def plot_corr_heatmap(df: pd.DataFrame):
    """数值特征相关性热力图"""
    num_cols = ["total_price_wan", "area_m2", "house_age",
                "district_avg_unit_price", "floor_level_num"]
    exist_cols = [c for c in num_cols if c in df.columns]
    corr = df[exist_cols].corr()

    plt.figure(figsize=(9, 6))
    im = plt.imshow(corr, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(exist_cols)), exist_cols, rotation=45, ha="right")
    plt.yticks(range(len(exist_cols)), exist_cols)
    plt.title("数值特征相关性热力图")
    plt.tight_layout()
    plt.show()


def plot_true_vs_pred(y_test, y_pred, title="真实值 vs 预测值"):
    """真实价格 vs 预测价格散点图"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    max_v = max(max(y_test), max(y_pred))
    min_v = min(min(y_test), min(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("真实总价（万）")
    plt.ylabel("预测总价（万）")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_residual_hist(y_test, y_pred):
    """残差分布"""
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("残差（真实-预测，万）")
    plt.ylabel("频数")
    plt.title("残差分布")
    plt.tight_layout()
    plt.show()


def plot_rf_feature_importance(rf_model, num_cols, cat_cols):
    """随机森林特征重要性（粗略展示）"""
    model = rf_model.named_steps["model"]
    preprocessor = rf_model.named_steps["preprocess"]

    # 数值列名
    num_features = num_cols

    # 类别特征展开后的 one-hot 名称
    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    all_feature_names = num_features + list(cat_feature_names)
    importances = model.feature_importances_

    # 只画前 15 个
    idx = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), [all_feature_names[i] for i in idx], rotation=45, ha="right")
    plt.ylabel("重要性")
    plt.title("随机森林特征重要性（Top 15）")
    plt.tight_layout()
    plt.show()


# ================== 主流程 ==================
def main():
    df = load_data()
    df = basic_clean(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test(df)
    preprocessor = build_preprocess_pipeline(num_cols, cat_cols)

    lin_reg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    rf_reg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    y_pred_test_lin = evaluate_model(
        lin_reg, X_train, X_test, y_train, y_test, name="LinearRegression"
    )
    y_pred_test_rf = evaluate_model(
        rf_reg, X_train, X_test, y_train, y_test, name="RandomForest"
    )

    # ====== 多张图 ======
    plot_price_distribution(df)
    plot_unit_price_distribution(df)
    plot_unit_price_by_district(df)
    plot_corr_heatmap(df)
    plot_true_vs_pred(y_test, y_pred_test_lin, title="线性回归：真实值 vs 预测值")
    plot_residual_hist(y_test, y_pred_test_lin)
    plot_true_vs_pred(y_test, y_pred_test_rf, title="随机森林：真实值 vs 预测值")
    plot_residual_hist(y_test, y_pred_test_rf)
    plot_rf_feature_importance(rf_reg, num_cols, cat_cols)


if __name__ == "__main__":
    main()
