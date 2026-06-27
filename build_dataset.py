"""
build_dataset.py  —  第二層：模型 + 資料整理
================================================
輸入：原始 5 個 CSV（DimCustomer / DimProduct / DimDate / FactHoldings / FactMonthlyValue）
動作：① 訓練財管傾向模型並對每位客戶評分
      ② 把儀表板需要的所有客戶明細整理乾淨（含 ML 分數與衍生欄位）
輸出：
      dashboard_data.csv   ← 客戶明細主檔（每位客戶一列，儀表板主要讀這份）
      dashboard_trend.csv  ← 月度趨勢（每客群每月一列，畫趨勢線用）

執行：
      pip install -r requirements-train.txt
      python build_dataset.py
"""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

SEGS = ["大眾理財", "財富管理", "貴賓理財", "私人銀行"]
SHORT = ["存", "卡", "貸", "財", "保"]
FLAGS = ["Has_Deposit", "Has_CreditCard", "Has_Loan", "Has_Wealth", "Has_Insurance"]

rd = lambda n: pd.read_csv(n, encoding="utf-8-sig")
cust = rd("DimCustomer.csv")
mv = rd("FactMonthlyValue.csv")

# ---------------- ① 訓練財管傾向模型 ----------------
y = cust["Has_Wealth"].astype(int)
num = ["Age", "TenureYears", "CardMonthlySpend_k", "LoanBalance_k"]
binf = ["Active90D", "Has_CreditCard", "Has_Loan", "Has_Insurance"]
catf = ["Segment", "IncomeBand", "Region", "Channel", "RiskProfile"]
X = cust[num + binf + catf]
pre = ColumnTransformer([("num", StandardScaler(), num + binf),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), catf)])
candidates = {
    "Logistic Regression": Pipeline([("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Gradient Boosting": Pipeline([("pre", pre),
        ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                               max_iter=300, random_state=42))]),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
best, best_auc, best_name = None, -1, ""
print("=" * 50, "\n 財管傾向模型評估\n", "=" * 50, sep="")
for name, m in candidates.items():
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    cv = cross_val_score(m, X, y, cv=5, scoring="roc_auc").mean()
    print(f"  {name:<20} 測試AUC={auc:.3f}  CV-AUC={cv:.3f}")
    if auc > best_auc:
        best, best_auc, best_name = m, auc, name
best.fit(X, y)
cust["WealthPropensity_ML"] = np.clip(np.round(best.predict_proba(X)[:, 1] * 100), 1, 99).astype(int)
print(f"  → 採用 {best_name}（AUC={best_auc:.3f}），已對 {len(cust)} 位客戶評分")

# ---------------- ② 整理客戶明細主檔 ----------------
cust["_seg"] = cust["Segment"].map({s: i for i, s in enumerate(SEGS)})
cust["AUM_萬"] = (cust["TotalAUM_k"] / 10).round().astype(int)
cust["現有商品"] = cust.apply(
    lambda r: "".join(SHORT[i] for i in range(5) if r[FLAGS[i]] == 1), axis=1)

keep = ["CustomerID", "Segment", "_seg", "Age", "AgeBand", "Region", "Channel",
        "TenureYears", "IncomeBand", "RiskProfile", "Active90D", "ProductCount",
        *FLAGS, "TotalAUM_k", "AUM_萬", "CardMonthlySpend_k", "LoanBalance_k",
        "AnnualContribution_k", "WealthPropensity", "WealthPropensity_ML", "現有商品"]
data = cust[keep]
data.to_csv("dashboard_data.csv", index=False, encoding="utf-8-sig")

# ---------------- 月度趨勢（每客群每月）----------------
trend = (mv.merge(cust[["CustomerID", "Segment"]], on="CustomerID")
           .groupby(["YearMonth", "Segment"], as_index=False)
           .agg(AUM_k=("AUM_k", "sum"), Contribution_k=("Contribution_k", "sum")))
trend.to_csv("dashboard_trend.csv", index=False, encoding="utf-8-sig")

print(f"\n✓ 已輸出 dashboard_data.csv（{len(data)} 列、{data.shape[1]} 欄）")
print(f"✓ 已輸出 dashboard_trend.csv（{len(trend)} 列）")
print("  接著執行： streamlit run app.py")
