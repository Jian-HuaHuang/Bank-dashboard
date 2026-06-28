"""
build_dataset.py  —  第二層：多商品傾向模型 + 資料整理
==========================================================
為「信用卡 / 放款 / 財富管理 / 保險」各訓練一個分類模型，預測客戶是否會持有該商品，
對每位客戶輸出 0–100 的傾向分數；並整理出儀表板所需的客戶明細與月度趨勢。

關鍵：每個商品的「資料洩漏」欄位不同，已逐一排除（見下方 leak_cols）。

輸入：DimCustomer.csv、FactMonthlyValue.csv
輸出：dashboard_data.csv（客戶明細，含 4 個商品傾向分數）
      dashboard_trend.csv（月度趨勢）
      model_metrics.csv（各商品模型表現）

執行： pip install -r requirements-train.txt  →  python build_dataset.py
"""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

SEGS = ["大眾理財", "財富管理", "貴賓理財", "私人銀行"]
SHORT = ["存", "卡", "貸", "財", "保"]
FLAGS = ["Has_Deposit", "Has_CreditCard", "Has_Loan", "Has_Wealth", "Has_Insurance"]

# 要建模的目標商品： 名稱 -> (目標旗標, 輸出分數欄位, 該商品專屬的洩漏欄位)
# 存款滲透率近 100%、無跨售空間，故不建模。
TARGETS = {
    "信用卡":   ("Has_CreditCard", "Prop_CreditCard", ["CardMonthlySpend_k"]),
    "放款":     ("Has_Loan",       "Prop_Loan",       ["LoanBalance_k"]),
    "財富管理": ("Has_Wealth",     "Prop_Wealth",     ["TotalAUM_k"]),
    "保險":     ("Has_Insurance",  "Prop_Insurance",  []),
}
# 對所有目標都排除（與目標高度相關、會洩漏答案）
ALWAYS_DROP_NUM = ["AnnualContribution_k"]          # 含多商品成分
ALWAYS_DROP_OTHER = ["ProductCount", "WealthPropensity"]  # 直接/間接洩漏

df = pd.read_csv("DimCustomer.csv", encoding="utf-8-sig")
mv = pd.read_csv("FactMonthlyValue.csv", encoding="utf-8-sig")

ALL_NUM = ["Age", "TenureYears", "CardMonthlySpend_k", "LoanBalance_k", "TotalAUM_k"]
ALL_CAT = ["Segment", "IncomeBand", "Region", "Channel", "RiskProfile"]
PROD_FLAGS = ["Has_CreditCard", "Has_Loan", "Has_Wealth", "Has_Insurance"]

FEAT_ZH = {
    "Age": "年齡", "TenureYears": "往來年資", "CardMonthlySpend_k": "信用卡月消費",
    "LoanBalance_k": "放款餘額", "TotalAUM_k": "AUM資產", "Active90D": "近90天活躍",
    "Has_CreditCard": "持有信用卡", "Has_Loan": "持有放款", "Has_Wealth": "持有財管",
    "Has_Insurance": "持有保險", "Segment": "客群", "IncomeBand": "收入級距",
    "Region": "區域", "Channel": "通路", "RiskProfile": "風險屬性",
}

metrics = []
roc_rows = []
imp_rows = []
print("=" * 60)
print(" 各商品傾向模型評估（已排除各自的資料洩漏欄位）")
print("=" * 60)

for name, (flag, outcol, leak) in TARGETS.items():
    y = df[flag].astype(int)
    num = [c for c in ALL_NUM if c not in leak]          # 排除該商品洩漏的金額欄
    binf = ["Active90D"] + [f for f in PROD_FLAGS if f != flag]  # 其他商品持有當特徵
    cat = ALL_CAT
    X = df[num + binf + cat]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num + binf),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    cands = {
        "LogReg": Pipeline([("pre", pre), ("clf",
            LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "GBoost": Pipeline([("pre", pre), ("clf",
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                           max_iter=300, random_state=42))]),
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    best, best_auc, best_name = None, -1, ""
    for mn, m in cands.items():
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        if auc > best_auc:
            best, best_auc, best_name = m, auc, mn
    cv = cross_val_score(best, X, y, cv=5, scoring="roc_auc").mean()

    # --- ROC 曲線（用held-out測試集，誠實）---
    proba_te = best.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba_te)
    idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)  # 抽稀以縮小檔案
    for f, tp in zip(fpr[idx], tpr[idx]):
        roc_rows.append({"商品": name, "fpr": round(float(f), 4), "tpr": round(float(tp), 4)})

    # --- 特徵重要度（permutation，模型無關、以原始特徵為單位）---
    perm = permutation_importance(best, Xte, yte, scoring="roc_auc",
                                  n_repeats=5, random_state=42)
    for col, imp in zip(X.columns, perm.importances_mean):
        imp_rows.append({"商品": name, "特徵": FEAT_ZH.get(col, col),
                         "重要度": round(float(max(imp, 0)), 4)})

    best.fit(X, y)
    df[outcol] = np.clip(np.round(best.predict_proba(X)[:, 1] * 100), 1, 99).astype(int)
    metrics.append({"商品": name, "模型": best_name,
                    "test_auc": round(best_auc, 3), "cv_auc": round(cv, 3),
                    "排除洩漏欄位": "、".join(leak) if leak else "（無專屬金額欄）"})
    print(f"  {name:<6} 採用 {best_name:<7} 測試AUC={best_auc:.3f}  CV={cv:.3f}  "
          f"排除洩漏：{leak if leak else '—'}")

# 向後相容：保留 WealthPropensity_ML = Prop_Wealth
df["WealthPropensity_ML"] = df["Prop_Wealth"]

# ---------- 整理客戶明細主檔 ----------
df["_seg"] = df["Segment"].map({s: i for i, s in enumerate(SEGS)})
df["AUM_萬"] = (df["TotalAUM_k"] / 10).round().astype(int)
df["現有商品"] = df.apply(lambda r: "".join(SHORT[i] for i in range(5) if r[FLAGS[i]] == 1), axis=1)

keep = (["CustomerID", "Segment", "_seg", "Age", "AgeBand", "Region", "Channel",
         "TenureYears", "IncomeBand", "RiskProfile", "Active90D", "ProductCount", *FLAGS,
         "TotalAUM_k", "AUM_萬", "CardMonthlySpend_k", "LoanBalance_k", "AnnualContribution_k"]
        + [t[1] for t in TARGETS.values()] + ["WealthPropensity_ML", "現有商品"])
df[keep].to_csv("dashboard_data.csv", index=False, encoding="utf-8-sig")

trend = (mv.merge(df[["CustomerID", "Segment"]], on="CustomerID")
           .groupby(["YearMonth", "Segment"], as_index=False)
           .agg(AUM_k=("AUM_k", "sum"), Contribution_k=("Contribution_k", "sum")))
trend.to_csv("dashboard_trend.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(metrics).to_csv("model_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(roc_rows).to_csv("model_roc.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(imp_rows).to_csv("model_importance.csv", index=False, encoding="utf-8-sig")

print("-" * 60)
print(f"✓ dashboard_data.csv（{len(df)} 列、{len(keep)} 欄，含 4 個商品傾向分數）")
print(f"✓ dashboard_trend.csv（{len(trend)} 列）")
print(f"✓ model_metrics.csv / model_roc.csv / model_importance.csv（模型洞察用）")
print("  接著執行： streamlit run app.py")
