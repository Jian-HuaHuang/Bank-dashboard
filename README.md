# 顧客跨商品儀表板 — 三層資料流架構

原始資料 → 模型＋整理程式 → 整理好的資料 → 儀表板。職責分層清楚，正是業界 pipeline 做法。

```
第一層  原始資料（訓練輸入）
        DimCustomer / DimProduct / DimDate / FactHoldings / FactMonthlyValue（5 個 CSV）
                 │
第二層  build_dataset.py  ─ 訓練財管傾向模型 + 評分 + 整理所有衍生欄位
                 │           輸出 ▼
        dashboard_data.csv（客戶明細主檔，含 ML 分數）
        dashboard_trend.csv（月度趨勢）
                 │
第三層  app.py（Streamlit）─ 只讀上面兩份檔，負責畫圖與客群篩選
```

## 怎麼跑（本機）
1. 產生整理檔（只在資料或模型更新時跑一次）：
   ```
   pip install -r requirements-train.txt
   python build_dataset.py
   ```
2. 啟動儀表板：
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```

## 部署到 Streamlit Cloud
- 只要 repo 內含 `app.py`、`dashboard_data.csv`、`dashboard_trend.csv`、`requirements.txt` 即可運作；
  **雲端不需要 scikit-learn**（分數已預先算好寫入 CSV）。
- 原始 5 個 CSV 與 `build_dataset.py` 放著做「可重現的 pipeline」佐證即可，部署用不到。

## 模型說明（面試可講）
- 目標：預測客戶是否持有財富管理（Has_Wealth）；對未持有者輸出傾向機率＝跨售分數。
- 特徵：客群、收入、年齡、年資、活躍度、其他商品持有、信用卡消費、放款餘額。
- 刻意排除資料洩漏欄位：TotalAUM_k、AnnualContribution_k、ProductCount。
- 表現：ROC-AUC ≈ 0.83（5 折交叉驗證 0.84）；採用 Logistic Regression（可解釋）。

## 設密碼（可選）
app.py 內建密碼關卡：到 Streamlit Cloud 的 Settings → Secrets 填
`app_password = "你的密碼"` 即生效；本機未設密碼時自動放行。

## 換成真實資料
保持欄位名稱不變，把第一層的原始 CSV 換成行內資料，重跑 build_dataset.py 即可，
app.py 完全不用改。真實場景下傾向分數應以歷史成交資料訓練模型。
