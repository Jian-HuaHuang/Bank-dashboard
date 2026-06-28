# 顧客跨商品分析儀表板 — 多商品傾向模型版（三層架構）

原始資料 → 多商品模型＋整理程式 → 整理好的資料 → 儀表板。

```
第一層  原始資料（5 個 CSV）
第二層  build_dataset.py ─ 為 4 個商品各訓練一個傾向模型 + 評分 + 整理
            輸出 ▼ dashboard_data.csv（含 4 個商品傾向分數）
                  dashboard_trend.csv（月度趨勢）
                  model_metrics.csv（各商品模型表現）
第三層  app.py（Streamlit）─ 只讀整理檔，畫圖與篩選
```

## 每一層的分析目標
- **第一層**：保存最原始、可重現的資料來源。
- **第二層**：把「原始事實」轉成「決策可用的明細」——核心是為每個商品建立傾向模型，回答「誰最可能接受這個商品」。
- **第三層**：將資料轉成商業可讀的畫面，回答三件事：客戶是誰、如何跨商品持有、跨售商機在哪。

## 執行
```
pip install -r requirements-train.txt
python build_dataset.py        # 訓練 4 個模型並產生整理檔（資料更新時才需重跑）
pip install -r requirements.txt
streamlit run app.py
```

## 多商品傾向模型
- 目標：分別預測客戶是否持有「信用卡 / 放款 / 財富管理 / 保險」。
- 每個商品**各自排除會洩漏答案的欄位**：預測信用卡排除信用卡消費、預測放款排除放款餘額、預測財管排除 AUM；所有模型一律排除 ProductCount、年度貢獻、原始示意分數。
- 各商品表現見 model_metrics.csv（畫面也會顯示）。財管訊號最強（AUC≈0.83）；放款在此合成資料近乎隨機（AUC≈0.54），屬資料本身特性，真實資料加入行為特徵可改善。
- 部署到 Streamlit Cloud 不需 scikit-learn：分數已預先算好寫入 CSV。

## 設密碼（可選）
Streamlit Cloud → Settings → Secrets 填 `app_password = "你的密碼"`。
