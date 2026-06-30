"""
app.py  —  第三層：儀表板（只讀整理好的資料）
================================================
只讀 build_dataset.py 產出的檔案：
    dashboard_data.csv    客戶明細主檔（含 4 商品 ML 傾向分數與衍生欄位）
    dashboard_trend.csv   月度趨勢（每客群每月一列）
    model_metrics.csv     各商品模型表現（AUC 等）
    model_roc.csv         各商品 ROC 曲線座標
    model_importance.csv  各商品特徵重要度
本程式不訓練模型、不碰原始 CSV，只負責「畫圖」與「篩選」。

執行：
    pip install -r requirements.txt
    streamlit run app.py
（若尚未產生整理檔，請先： python build_dataset.py）
"""
import os                                  # 處理檔案路徑
import numpy as np                         # 數值運算（矩陣用）
import pandas as pd                        # 讀 CSV、資料處理
import streamlit as st                     # 網頁儀表板框架
import plotly.express as px                # 快速畫圖（長條、圓餅、熱力圖）
import plotly.graph_objects as go          # 進階畫圖（雙軸、ROC 曲線）
from plotly.subplots import make_subplots  # 建立雙 Y 軸圖

# 頁面基本設定：分頁標題、圖示、寬版面
st.set_page_config(page_title="顧客跨商品儀表板", page_icon="🟥", layout="wide")

# ── 永豐品牌色票（給圖表與 CSS 使用）──
MAROON, RED, GOLD = "#5C1A14", "#C4392C", "#C0982F"        # 深紅、正紅、金
SOFT, DARK2, SAND, CREAM = "#E07A5F", "#7A2018", "#E8C9A0", "#F2DEB0"  # 輔助色
INK, BG, MUTED = "#2B1714", "#F6EDE9", "#8C7068"           # 文字、背景、次要文字
SEQ = [RED, MAROON, GOLD, SOFT, DARK2]                     # 類別配色順序（圓餅/分組柱）
HEAT = [[0, "#FBE3DC"], [0.5, RED], [1.0, MAROON]]         # 熱力圖漸層（淺→深紅）

# ── 五大商品類別（順序固定，後面矩陣與旗標都依此順序）──
CATS = ["存款", "信用卡", "放款", "財富管理", "保險"]
# 對應 CATS 的「是否持有」旗標欄位（1=持有、0=未持有），與 CATS 一一對應
FLAGS = ["Has_Deposit", "Has_CreditCard", "Has_Loan", "Has_Wealth", "Has_Insurance"]
# 客群分級（由低到高）
SEGS = ["大眾理財", "財富管理", "貴賓理財", "私人銀行"]

# ── 可建模跨售的目標商品（存款近 100% 滲透、無跨售空間，故不列入）──
TARGET_NAMES = ["信用卡", "放款", "財富管理", "保險"]
# 目標商品 → 對應的「是否持有」旗標欄位
TARGET_FLAG = {"信用卡": "Has_CreditCard", "放款": "Has_Loan",
               "財富管理": "Has_Wealth", "保險": "Has_Insurance"}
# 目標商品 → 對應的「ML 傾向分數」欄位（由 build_dataset.py 算好寫入）
TARGET_COL = {"信用卡": "Prop_CreditCard", "放款": "Prop_Loan",
              "財富管理": "Prop_Wealth", "保險": "Prop_Insurance"}

# ── 自訂 CSS：背景、側邊欄、指標卡的外觀 ──
st.markdown(f"""
<style>
.stApp {{ background:{BG}; }}
section[data-testid="stSidebar"] > div {{ background:{MAROON}; }}
section[data-testid="stSidebar"] * {{ color:#F3E1DB !important; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #ECD8D2; border-radius:14px;
    padding:14px 16px; box-shadow:0 2px 8px rgba(67,16,9,.05); }}
div[data-testid="stMetricValue"] {{ color:{MAROON}; font-weight:800; }}
div[data-testid="stMetricLabel"] p {{ color:{MUTED}; font-size:13px; }}
h1,h2,h3 {{ color:{MAROON}; }}
</style>
""", unsafe_allow_html=True)


# ---------------- 密碼保護 ----------------
def check_password():
    # 若本次連線已通過驗證，直接放行
    if st.session_state.get("auth_ok"):
        return True

    # 比對使用者輸入的密碼是否正確
    def _verify():
        try:
            real = st.secrets["app_password"]          # 從 Streamlit Secrets 取設定的密碼
        except Exception:
            st.session_state["auth_ok"] = True         # 未設定密碼時不阻擋（方便本機測試）
            return
        # 輸入值與正確密碼相符才放行
        st.session_state["auth_ok"] = (st.session_state.get("pw_input", "") == real)

    # 顯示密碼輸入框
    st.markdown(f"<h2 style='color:{MAROON}'>🔒 顧客跨商品儀表板</h2>", unsafe_allow_html=True)
    st.text_input("請輸入存取密碼", type="password", key="pw_input", on_change=_verify)
    if st.session_state.get("auth_ok") is False:
        st.error("密碼錯誤，請再試一次。")
    st.caption("此儀表板僅供授權對象檢視。")
    return False


# 未通過密碼驗證就停止後續所有畫面
if not check_password():
    st.stop()


# ---------------- 載入整理好的資料 ----------------
@st.cache_data                              # 快取：同一份資料只讀一次，加速
def load(d="."):
    # 主檔：客戶明細（每位客戶一列）
    data = pd.read_csv(os.path.join(d, "dashboard_data.csv"), encoding="utf-8-sig")
    # 趨勢檔：每客群每月的 AUM 與貢獻
    trend = pd.read_csv(os.path.join(d, "dashboard_trend.csv"), encoding="utf-8-sig")

    # 模型相關檔案可能不存在，讀不到就回傳 None（不讓程式崩潰）
    def opt(fn):
        try:
            return pd.read_csv(os.path.join(d, fn), encoding="utf-8-sig")
        except Exception:
            return None

    # 依序回傳：主檔、趨勢、模型表現、ROC、特徵重要度
    return data, trend, opt("model_metrics.csv"), opt("model_roc.csv"), opt("model_importance.csv")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))    # app.py 所在資料夾
try:
    # 一次載入五份資料
    DATA, TREND, METRICS, ROC, IMP = load(DATA_DIR)
except Exception as e:
    # 找不到整理檔時，提示先跑 build_dataset.py
    st.error(f"找不到整理檔。請先在本資料夾執行： python build_dataset.py\n\n錯誤：{e}")
    st.stop()


# ---------------- 側邊欄（導覽與篩選）----------------
with st.sidebar:
    # 品牌標題
    st.markdown('<div class="brand">🟥 <b>永豐銀行</b><br><span>顧客跨商品儀表板</span></div>',
                unsafe_allow_html=True)
    st.markdown("---")
    # 分析頁面切換（四頁）
    page = st.radio("分析頁面", ["① 經營總覽", "② 跨商品行為", "③ 跨售商機", "④ 模型洞察"])
    st.markdown("---")
    # 客群篩選器（會連動前三頁的計算）
    seg_sel = st.selectbox("客群篩選", ["全部客群"] + SEGS)
    st.caption("資料為符合真實分布之合成示意資料 ‧ 非真實客戶\n傾向分數由機器學習模型產生 (AUC≈0.83)")

# C＝套用客群篩選後的客戶資料；選「全部客群」則用全部，否則只留該客群
C = DATA if seg_sel == "全部客群" else DATA[DATA["Segment"] == seg_sel]
n = len(C)                                  # 篩選後的客戶數（後面多處共用）
pct = lambda x: f"{x:.1f}%"                 # 小工具：數字轉成「xx.x%」字串


# ================= 頁面 1：經營總覽 =================
if page.startswith("①"):
    st.title("經營總覽")
    st.caption(f"目前客群：{seg_sel}　｜　客群結構、商品滲透與貢獻全貌")

    # ── 五張關鍵指標（KPI）卡 ──
    k = st.columns(5)
    k[0].metric("總客戶數", f"{n:,}")                                  # 客戶總數
    k[1].metric("平均產品密度", f"{C.ProductCount.mean():.2f}")        # 平均每人持有幾種商品
    k[2].metric("跨售率 ≥2 項", pct((C.ProductCount >= 2).mean() * 100))  # 持有≥2項的人占比
    k[3].metric("總資產規模 AUM", f"{C.TotalAUM_k.sum()/1e5:.1f} 億")   # AUM總額(千元)÷10萬=億
    # 月活躍率＝Active90D(近90天有往來=1)的平均值×100，即「活躍客戶占比」
    k[4].metric("月活躍率", pct(C.Active90D.mean() * 100))

    c1, c2 = st.columns([1.2, .8])
    # ── 左：月度 AUM(柱) 與 貢獻(折線) 雙軸趨勢 ──
    with c1:
        # 依客群篩選趨勢資料
        tt = TREND if seg_sel == "全部客群" else TREND[TREND.Segment == seg_sel]
        # 把同月份不同客群加總成每月一個值
        tr = tt.groupby("YearMonth", as_index=False).agg(AUM=("AUM_k", "sum"), CO=("Contribution_k", "sum"))
        tr = tr.sort_values("YearMonth")                  # 依年月排序，避免折線亂跳
        fig = make_subplots(specs=[[{"secondary_y": True}]])  # 建立左右雙 Y 軸
        # 柱：AUM（千元→億），並在柱上標數值
        fig.add_bar(x=tr.YearMonth, y=tr.AUM/1e5, name="AUM(億)", marker_color=RED,
                    text=tr.AUM/1e5, texttemplate="%{text:.1f}", textposition="outside",
                    textfont=dict(size=9, color=MAROON), cliponaxis=False)
        # 折線：貢獻（千元→萬，÷10），畫在第二 Y 軸，並在點上標數值
        fig.add_trace(go.Scatter(x=tr.YearMonth, y=tr.CO/10, name="貢獻(萬)",
                                 mode="lines+markers+text", text=(tr.CO/10).round().astype(int),
                                 texttemplate="%{text}", textposition="top center",
                                 textfont=dict(size=9, color=GOLD),
                                 line=dict(color=GOLD, width=3)), secondary_y=True)
        fig.update_layout(title="月度 AUM 與貢獻度", height=330,
                          margin=dict(t=40, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=-0.2), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    # ── 右：客群結構環圈圖 ──
    with c2:
        # 計算各客群人數，並依 SEGS 順序排列
        sc = C.Segment.value_counts().reindex(SEGS).fillna(0).reset_index()
        sc.columns = ["Segment", "n"]                     # 欄位改名：客群、人數
        fig = px.pie(sc, names="Segment", values="n", hole=.6,
                     color_discrete_sequence=SEQ, title="客群結構")
        fig.update_traces(textinfo="percent", textfont_size=12)  # 每塊標示百分比
        fig.update_layout(height=330, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    # ── 左下：往來通路結構（橫條）──
    with c3:
        ch = C.Channel.value_counts().reset_index(); ch.columns = ["Channel", "n"]  # 各通路人數
        fig = px.bar(ch, x="n", y="Channel", orientation="h",
                     color_discrete_sequence=[DARK2], title="往來通路結構")
        # 在條尾標示人數（千分位）
        fig.update_traces(texttemplate="%{x:,}", textposition="outside",
                          cliponaxis=False, textfont_size=11)
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=40, r=40), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    # ── 右下：各商品滲透率（橫條）──
    with c4:
        # 對每個商品旗標取平均×100＝該商品的持有率(%)
        dfp = pd.DataFrame({"商品": CATS, "滲透率": [C[f].mean()*100 for f in FLAGS]})
        fig = px.bar(dfp, x="滲透率", y="商品", orientation="h",
                     color="商品", color_discrete_sequence=SEQ, title="各商品滲透率")
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside",
                          cliponaxis=False, textfont_size=11)
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=40),
                          showlegend=False, plot_bgcolor="white", xaxis_range=[0, 112])
        st.plotly_chart(fig, use_container_width=True)


# ================= 頁面 2：跨商品行為 =================
elif page.startswith("②"):
    st.title("跨商品行為")
    st.caption(f"目前客群：{seg_sel}　｜　客戶在五大商品線的交叉持有樣態")

    c1, c2 = st.columns([.85, 1.15])
    # ── 左：客戶產品密度分布（持有 0~5 種商品各多少人）──
    with c1:
        # 統計每種「持有商品數」的人數，補滿 0~5
        dh = C.ProductCount.value_counts().reindex(range(6)).fillna(0).reset_index()
        dh.columns = ["持有商品數", "客戶數"]
        fig = px.bar(dh, x="持有商品數", y="客戶數", title="客戶產品密度分布")
        # 持有<2 種用淺色(待加強)、≥2 用紅色(已跨售)，並標人數
        fig.update_traces(marker_color=[SAND if v < 2 else RED for v in dh["持有商品數"]],
                          texttemplate="%{y:,}", textposition="outside",
                          cliponaxis=False, textfont_size=11)
        fig.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    # ── 右：跨商品持有矩陣（列＝已持有，欄＝同時持有比率%）──
    with c2:
        mat = np.zeros((5, 5))                            # 5×5 矩陣，預設 0
        for r in range(5):                               # r＝列：已持有的商品
            base = C[C[FLAGS[r]] == 1]                    # 取出「持有第 r 種商品」的客戶
            for cc in range(5):                          # cc＝欄：另一個商品
                # 這群人中也持有第 cc 種商品的比例(%)；分母為 0 時填 0
                mat[r, cc] = base[FLAGS[cc]].mean()*100 if len(base) else 0
        fig = px.imshow(mat, x=CATS, y=CATS, color_continuous_scale=HEAT,
                        text_auto=".0f", aspect="auto",        # 每格直接顯示數字
                        title="跨商品持有矩陣（列＝已持有，欄＝同時持有比率 %）")
        fig.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 下：各客群商品持有率（分組柱）──
    # 對每個客群×每個商品算持有率(%)（用全體 DATA，不受篩選影響，方便客群互相比較）
    rows = [{"客群": s, "商品": CATS[ci], "滲透率": DATA[DATA.Segment == s][FLAGS[ci]].mean()*100}
            for s in SEGS for ci in range(5)]
    fig = px.bar(pd.DataFrame(rows), x="客群", y="滲透率", color="商品", barmode="group",
                 color_discrete_sequence=SEQ, title="各客群商品持有率")
    fig.update_traces(texttemplate="%{y:.0f}", textposition="outside",
                      cliponaxis=False, textfont_size=8)
    fig.update_layout(height=360, margin=dict(t=40, b=10, l=10, r=10),
                      plot_bgcolor="white", yaxis_range=[0, 112])
    st.plotly_chart(fig, use_container_width=True)


# ================= 頁面 3：跨售商機 =================
elif page.startswith("③"):
    st.title("跨售商機")
    st.caption(f"目前客群：{seg_sel}　｜　鎖定潛在客戶，量化跨售與向上銷售潛力")

    # 選擇要推廣的目標商品（預設財富管理）
    t_name = st.radio("推廣目標商品", TARGET_NAMES, index=TARGET_NAMES.index("財富管理"), horizontal=True)
    flag = TARGET_FLAG[t_name]                    # 該商品的「是否持有」旗標欄
    pcol = TARGET_COL[t_name]                     # 該商品的「ML 傾向分數」欄
    # 潛在客戶＝尚未持有此商品(flag==0) 且 至少持有一種商品(ProductCount>=1) 的人
    ws = C[(C[flag] == 0) & (C.ProductCount >= 1)].copy()
    ws["score"] = ws[pcol]                         # 取出該商品的傾向分數當作排序依據

    # 顯示此商品模型的來源與 AUC；AUC<0.6 標註「訊號較弱」
    if METRICS is not None and (METRICS["商品"] == t_name).any():
        r = METRICS[METRICS["商品"] == t_name].iloc[0]
        warn = "（訊號較弱，排序參考用）" if r["test_auc"] < 0.6 else ""
        st.caption(f"{t_name}傾向分數來源：機器學習模型（{r['模型']}，測試 AUC≈{r['test_auc']:.2f}）{warn}")
    else:
        st.caption(f"{t_name}傾向分數來源：機器學習模型")

    hi = int((ws.score >= 60).sum())              # 優先推薦客戶數＝傾向分數≥60 的人數
    with_t = C[C[flag] == 1].AnnualContribution_k     # 已持有此商品者的年貢獻
    without_t = C[C[flag] == 0].AnnualContribution_k  # 未持有此商品者的年貢獻
    # 貢獻落差＝有持有者平均貢獻 − 未持有者平均貢獻（不為負）
    uplift = max(0, (with_t.mean() if len(with_t) else 0) - (without_t.mean() if len(without_t) else 0))
    pot = hi * uplift / 10                         # 估計年貢獻潛力(萬)＝高傾向人數×貢獻落差÷10

    # ── 四張商機指標卡 ──
    k = st.columns(4)
    k[0].metric(f"潛在客戶（未持有{t_name}）", f"{len(ws):,}")        # 潛在客戶數
    k[1].metric("優先推薦客戶（分數≥60）", f"{hi:,}")                # 高傾向人數
    k[2].metric("潛在客戶占比", pct(100*len(ws)/n if n else 0))      # 潛在客戶佔篩選範圍比例
    k[3].metric("估計年貢獻潛力", f"{pot:,.0f} 萬")                  # 商機金額(示意)

    c1, c2 = st.columns(2)
    # ── 左：潛在客戶傾向分數分布（分 5 個區間）──
    with c1:
        # 把分數切成 0-20…80-100 五段並計數
        b = pd.cut(ws.score, [0, 20, 40, 60, 80, 100],
                   labels=["0-20", "20-40", "40-60", "60-80", "80-100"]).value_counts().sort_index().reset_index()
        b.columns = ["區間", "客戶數"]
        fig = px.bar(b, x="區間", y="客戶數", color="區間", title="潛在客戶傾向分數分布",
                     color_discrete_sequence=[SAND, SOFT, GOLD, RED, MAROON])
        fig.update_traces(texttemplate="%{y:,}", textposition="outside",
                          cliponaxis=False, textfont_size=11)
        fig.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    # ── 右：跨售缺口（持有列、未持有欄的客戶數）──
    with c2:
        gap = np.zeros((5, 5))
        for r in range(5):                            # r＝列：已持有的商品
            base = C[C[FLAGS[r]] == 1]                # 持有第 r 種商品的客戶
            for cc in range(5):                       # cc＝欄：另一個商品
                # 對角線(自己對自己)填 0；其餘＝這群人中「未持有第 cc 種」的人數
                gap[r, cc] = 0 if r == cc else int((base[FLAGS[cc]] == 0).sum())
        fig = px.imshow(gap, x=CATS, y=CATS, color_continuous_scale=HEAT,
                        text_auto=".0f", aspect="auto",
                        title="跨售缺口（持有列、未持有欄的客戶數）")
        fig.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 優先推薦名單（依傾向分數由高到低取前 40）──
    st.subheader("優先推薦名單（依傾向分數由高到低，取前 40）")
    top = ws.sort_values("score", ascending=False).head(40).copy()  # 取分數最高的 40 人
    top["建議商品"] = t_name                                        # 標註建議推廣的商品
    # 只挑要顯示的欄位，並把欄名改成中文
    show = top[["CustomerID", "Segment", "Age", "TenureYears", "AUM_萬",
                "現有商品", "建議商品", "score"]].rename(
        columns={"CustomerID": "客戶ID", "Segment": "客群", "Age": "年齡",
                 "TenureYears": "往來年資", "AUM_萬": "AUM(萬)", "score": "傾向分數"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)


# ================= 頁面 4：模型洞察 =================
else:
    st.title("模型洞察")
    st.caption("各商品傾向模型的表現、ROC 曲線與特徵重要度（資料科學佐證）")

    # ── 各商品模型表現表（AUC、CV-AUC 等）──
    if METRICS is not None:
        st.subheader("各商品模型表現")
        mt = METRICS.rename(columns={"test_auc": "測試AUC", "cv_auc": "CV-AUC"})  # 欄名中文化
        st.dataframe(mt, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    # ── 左：四商品 ROC 曲線疊圖 ──
    with c1:
        st.subheader("ROC 曲線")
        if ROC is not None:
            fig = go.Figure()
            palette = {"信用卡": RED, "放款": GOLD, "財富管理": MAROON, "保險": SOFT}  # 各商品線色
            for nm in TARGET_NAMES:
                sub = ROC[ROC["商品"] == nm].sort_values("fpr")   # 取該商品的曲線座標
                auc = ""
                # 圖例標上該商品 AUC
                if METRICS is not None and (METRICS["商品"] == nm).any():
                    auc = f"（AUC {METRICS[METRICS['商品']==nm]['test_auc'].iloc[0]:.2f}）"
                fig.add_trace(go.Scatter(x=sub.fpr, y=sub.tpr, mode="lines",
                                         name=f"{nm}{auc}", line=dict(color=palette.get(nm), width=2.5)))
            # 對角線＝隨機猜測的基準線
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="隨機",
                                     line=dict(color="#BBB", dash="dash", width=1)))
            fig.update_layout(height=380, margin=dict(t=20, b=10, l=10, r=10),
                              xaxis_title="假陽性率 (FPR)", yaxis_title="真陽性率 (TPR)",
                              legend=dict(font=dict(size=10)), plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("曲線越靠左上、AUC 越大＝模型越能區分會買與不會買的客戶。")
        else:
            st.info("尚無 ROC 資料，請重新執行 build_dataset.py。")
    # ── 右：特徵重要度（可切換商品）──
    with c2:
        st.subheader("特徵重要度")
        prod = st.selectbox("選擇商品", TARGET_NAMES, index=TARGET_NAMES.index("財富管理"))
        if IMP is not None:
            # 取該商品的特徵重要度，由小到大排序（橫條由下往上）
            sub = IMP[IMP["商品"] == prod].sort_values("重要度", ascending=True)
            sub = sub[sub["重要度"] > 0].tail(10)         # 只留重要度>0 的前 10 名
            fig = px.bar(sub, x="重要度", y="特徵", orientation="h",
                         color_discrete_sequence=[RED])
            fig.update_traces(texttemplate="%{x:.3f}", textposition="outside",
                              cliponaxis=False, textfont_size=10)
            fig.update_layout(height=380, margin=dict(t=20, b=10, l=10, r=60), plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("數值＝該特徵被打亂後 AUC 的下降幅度（permutation importance）；越大代表越關鍵。")
        else:
            st.info("尚無特徵重要度資料，請重新執行 build_dataset.py。")

    # ── 本頁解讀說明 ──
    st.markdown("---")
    st.markdown(
        "**怎麼讀這頁**：ROC 曲線比較四個商品模型的整體鑑別力（財富管理最佳、放款最弱）；"
        "特徵重要度告訴你每個模型「主要靠什麼在判斷」。例如財富管理幾乎全靠『客群』，"
        "反映此合成資料中財管持有高度由客群決定；真實資料納入交易行為後，重要度會更分散、模型也更強。")
