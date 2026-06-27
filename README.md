# 銀行顧客跨商品儀表板 — Streamlit 版

用純 Python（Streamlit + Plotly）重現互動式儀表板，三頁：經營總覽 / 跨商品行為 / 跨售商機，
左側可篩選客群並即時連動全部圖表。

## 執行方式
1. 安裝套件（只需一次）：
   ```
   pip install -r requirements.txt
   ```
2. 在本資料夾（含 5 個 CSV 與 app.py）執行：
   ```
   streamlit run app.py
   ```
3. 瀏覽器會自動開啟 http://localhost:8501

## 需要的檔案（已附）
DimCustomer.csv ‧ DimProduct.csv ‧ DimDate.csv ‧ FactHoldings.csv ‧ FactMonthlyValue.csv
（app.py 會自動讀取與它同資料夾的這 5 個檔）

## 換成真實資料
保持欄位名稱不變，把 DimCustomer / FactHoldings / FactMonthlyValue 換成你的資料即可，程式不用改。
傾向分數（WealthPropensity）實務上以歷史成交資料訓練分類模型後寫回客戶表。

## 想分享給別人看？
可免費部署到 Streamlit Community Cloud（share.streamlit.io）：把這個資料夾推到 GitHub，
在平台選 app.py 即可得到一個公開網址。
