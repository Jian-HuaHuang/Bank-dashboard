"""
銀行顧客跨商品分析儀表板 — Streamlit 版
執行方式：
    pip install streamlit plotly pandas
    streamlit run app.py
需要與本檔同資料夾的 5 個 CSV：
    DimCustomer.csv, DimProduct.csv, DimDate.csv, FactHoldings.csv, FactMonthlyValue.csv
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- 設定與配色 ----------------
st.set_page_config(page_title="銀行顧客跨商品儀表板", page_icon="🟥", layout="wide")

MAROON, RED, GOLD = "#5C1A14", "#C4392C", "#C0982F"
SOFT, DARK2, SAND, CREAM = "#E07A5F", "#7A2018", "#E8C9A0", "#F2DEB0"
INK, BG, MUTED = "#2B1714", "#F6EDE9", "#8C7068"
SEQ = [RED, MAROON, GOLD, SOFT, DARK2]                 # 類別色序
HEAT = [[0, "#FBE3DC"], [0.5, RED], [1.0, MAROON]]     # 熱力圖色階

CATS = ["存款", "信用卡", "放款", "財富管理", "保險"]
FLAGS = ["Has_Deposit", "Has_CreditCard", "Has_Loan", "Has_Wealth", "Has_Insurance"]
SEGS = ["大眾理財", "財富管理", "貴賓理財", "私人銀行"]

st.markdown(f"""
<style>
.stApp {{ background:{BG}; }}
section[data-testid="stSidebar"] > div {{ background:{MAROON}; }}
section[data-testid="stSidebar"] * {{ color:#F3E1DB !important; }}
section[data-testid="stSidebar"] .brand b {{ font-size:18px; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #ECD8D2; border-radius:14px;
    padding:14px 16px; box-shadow:0 2px 8px rgba(67,16,9,.05); }}
div[data-testid="stMetricValue"] {{ color:{MAROON}; font-weight:800; }}
div[data-testid="stMetricLabel"] p {{ color:{MUTED}; font-size:13px; }}
h1,h2,h3 {{ color:{MAROON}; }}
hr {{ border-color:#ECD8D2; }}
</style>
""", unsafe_allow_html=True)

# ---------------- 密碼保護 ----------------
def check_password():
    """顯示密碼輸入框；正確才回傳 True。密碼存在 Streamlit Secrets 的 app_password。"""
    if st.session_state.get("auth_ok"):
        return True

    def _verify():
        try:
            real = st.secrets["app_password"]
        except Exception:
            st.session_state["auth_ok"] = True   # 未設定密碼時不阻擋（方便本機測試）
            return
        if st.session_state.get("pw_input", "") == real:
            st.session_state["auth_ok"] = True
            st.session_state["pw_input"] = ""
        else:
            st.session_state["auth_ok"] = False

    st.markdown(f"<h2 style='color:{MAROON}'>🔒 永豐顧客跨商品儀表板</h2>", unsafe_allow_html=True)
    st.text_input("請輸入存取密碼", type="password", key="pw_input", on_change=_verify)
    if st.session_state.get("auth_ok") is False:
        st.error("密碼錯誤，請再試一次。")
    st.caption("此儀表板僅供授權對象檢視。")
    return False


if not check_password():
    st.stop()

# ---------------- 載入資料 ----------------
@st.cache_data
def load(data_dir="."):
    f = lambda n: pd.read_csv(os.path.join(data_dir, n))
    cust = f("DimCustomer.csv")
    hold = f("FactHoldings.csv")
    mv = f("FactMonthlyValue.csv")
    seg_code = {s: i for i, s in enumerate(SEGS)}
    cust["_seg"] = cust["Segment"].map(seg_code)
    return cust, hold, mv

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    cust, hold, mv = load(DATA_DIR)
except Exception as e:
    st.error(f"找不到 CSV，請把 5 個 CSV 放在與 app.py 相同資料夾。錯誤：{e}")
    st.stop()

# ---------------- 側邊欄 ----------------
with st.sidebar:
    st.markdown('<div class="brand">🟥 <b>永豐銀行</b><br><span>顧客跨商品儀表板</span></div>',
                unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("分析頁面", ["① 經營總覽", "② 跨商品行為", "③ 跨售商機"])
    st.markdown("---")
    seg_sel = st.selectbox("客群篩選", ["全部客群"] + SEGS)
    st.caption("資料為符合真實分布之合成示意資料 ‧ 非真實客戶\nn = 5,000 ‧ 2025/07–2026/06")

# 套用客群篩選
if seg_sel == "全部客群":
    C = cust.copy()
    ids = None
else:
    C = cust[cust["Segment"] == seg_sel].copy()
    ids = set(C["CustomerID"])
H = hold if ids is None else hold[hold["CustomerID"].isin(ids)]
M = mv if ids is None else mv[mv["CustomerID"].isin(ids)]
n = len(C)

def pct(x): return f"{x:.1f}%"

# ================= 頁面 1：經營總覽 =================
if page.startswith("①"):
    st.title("經營總覽")
    st.caption(f"目前客群：{seg_sel}　｜　客群結構、商品滲透與貢獻全貌")

    density = C["ProductCount"].mean()
    cross = (C["ProductCount"] >= 2).mean() * 100
    aum_y = C["TotalAUM_k"].sum() / 1e5
    active = C["Active90D"].mean() * 100
    k = st.columns(5)
    k[0].metric("總客戶數", f"{n:,}")
    k[1].metric("平均產品密度", f"{density:.2f}")
    k[2].metric("跨售率 ≥2 項", pct(cross))
    k[3].metric("總資產規模 AUM", f"{aum_y:.1f} 億")
    k[4].metric("月活躍率", pct(active))

    c1, c2 = st.columns([1.2, .8])
    with c1:
        tr = (M.groupby("YearMonth")
                .agg(AUM=("AUM_k", "sum"), CO=("Contribution_k", "sum"))
                .reset_index().sort_values("YearMonth"))
        tr["AUM億"] = tr["AUM"] / 1e5
        tr["貢獻萬"] = tr["CO"] / 10
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=tr["YearMonth"], y=tr["AUM億"], name="AUM(億)", marker_color=RED)
        fig.add_trace(go.Scatter(x=tr["YearMonth"], y=tr["貢獻萬"], name="貢獻(萬)",
                                 line=dict(color=GOLD, width=3)), secondary_y=True)
        fig.update_layout(title="月度 AUM 與貢獻度", height=330,
                          margin=dict(t=40, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=-0.2), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sc = C["Segment"].value_counts().reindex(SEGS).fillna(0).reset_index()
        sc.columns = ["Segment", "n"]
        fig = px.pie(sc, names="Segment", values="n", hole=.6,
                     color_discrete_sequence=SEQ, title="客群結構")
        fig.update_layout(height=330, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        ch = C["Channel"].value_counts().reset_index()
        ch.columns = ["Channel", "n"]
        fig = px.bar(ch, x="n", y="Channel", orientation="h",
                     color_discrete_sequence=[DARK2], title="往來通路結構")
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        pen = [C[fl].mean() * 100 for fl in FLAGS]
        dfp = pd.DataFrame({"商品": CATS, "滲透率": pen})
        fig = px.bar(dfp, x="滲透率", y="商品", orientation="h",
                     color="商品", color_discrete_sequence=SEQ, title="各商品滲透率")
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10),
                          showlegend=False, plot_bgcolor="white", xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

# ================= 頁面 2：跨商品行為 =================
elif page.startswith("②"):
    st.title("跨商品行為")
    st.caption(f"目前客群：{seg_sel}　｜　客戶在五大商品線的交叉持有樣態")

    c1, c2 = st.columns([.85, 1.15])
    with c1:
        dh = C["ProductCount"].value_counts().reindex(range(6)).fillna(0).reset_index()
        dh.columns = ["持有商品數", "客戶數"]
        colors = [SAND if v < 2 else RED for v in dh["持有商品數"]]
        fig = px.bar(dh, x="持有商品數", y="客戶數", title="客戶產品密度分布")
        fig.update_traces(marker_color=colors)
        fig.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # 跨商品持有矩陣：P(欄 | 列)
        mat = np.zeros((5, 5))
        for r in range(5):
            base = C[C[FLAGS[r]] == 1]
            for cc in range(5):
                mat[r, cc] = (base[FLAGS[cc]].mean() * 100) if len(base) else 0
        fig = px.imshow(mat, x=CATS, y=CATS, color_continuous_scale=HEAT,
                        text_auto=".0f", aspect="auto",
                        title="跨商品持有矩陣（列＝已持有，欄＝同時持有比率 %）")
        fig.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # 各客群商品持有率
    rows = []
    for s in SEGS:
        sub = cust[cust["Segment"] == s]
        for ci, fl in enumerate(FLAGS):
            rows.append({"客群": s, "商品": CATS[ci], "滲透率": sub[fl].mean() * 100})
    dseg = pd.DataFrame(rows)
    fig = px.bar(dseg, x="客群", y="滲透率", color="商品", barmode="group",
                 color_discrete_sequence=SEQ, title="各客群商品持有率")
    fig.update_layout(height=360, margin=dict(t=40, b=10, l=10, r=10),
                      plot_bgcolor="white", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

# ================= 頁面 3：跨售商機 =================
else:
    st.title("跨售商機")
    st.caption(f"目前客群：{seg_sel}　｜　鎖定白地客戶，量化跨售與向上銷售潛力")

    t_name = st.radio("推廣目標商品", CATS, index=3, horizontal=True)
    t = CATS.index(t_name)

    ws = C[(C[FLAGS[t]] == 0) & (C["ProductCount"] >= 1)].copy()

    # 傾向分數：財富管理用真實分數，其餘用啟發式（示意）
    if t == 3:
        ws["score"] = ws["WealthPropensity"]
    else:
        z = (-1.0 + 0.5 * ws["_seg"] + 0.04 * (60 - (ws["Age"] - 46).abs())
             + 0.04 * ws["TenureYears"] + 1.6e-5 * ws["TotalAUM_k"] + 0.6 * ws["Active90D"])
        ws["score"] = (100 / (1 + np.exp(-z))).clip(1, 99).round()
    hi = int((ws["score"] >= 60).sum())

    with_t = C[C[FLAGS[t]] == 1]["AnnualContribution_k"]
    without_t = C[C[FLAGS[t]] == 0]["AnnualContribution_k"]
    uplift = max(0, (with_t.mean() if len(with_t) else 0) - (without_t.mean() if len(without_t) else 0))
    pot_wan = hi * uplift / 10

    k = st.columns(4)
    k[0].metric(f"白地客戶（無{t_name}）", f"{len(ws):,}")
    k[1].metric("高傾向客戶（≥60）", f"{hi:,}")
    k[2].metric("白地占比", pct(100 * len(ws) / n if n else 0))
    k[3].metric("估計年貢獻潛力", f"{pot_wan:,.0f} 萬")

    c1, c2 = st.columns(2)
    with c1:
        bins = pd.cut(ws["score"], [0, 20, 40, 60, 80, 100],
                      labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
        bh = bins.value_counts().sort_index().reset_index()
        bh.columns = ["區間", "客戶數"]
        fig = px.bar(bh, x="區間", y="客戶數", title="白地客戶傾向分數分布",
                     color="區間",
                     color_discrete_sequence=[SAND, SOFT, GOLD, RED, MAROON])
        fig.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10),
                          showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gap = np.zeros((5, 5))
        for r in range(5):
            base = C[C[FLAGS[r]] == 1]
            for cc in range(5):
                gap[r, cc] = 0 if r == cc else int((base[FLAGS[cc]] == 0).sum())
        fig = px.imshow(gap, x=CATS, y=CATS, color_continuous_scale=HEAT,
                        text_auto=".0f", aspect="auto",
                        title="跨售白地缺口（有列、缺欄的客戶數）")
        fig.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("重點跨售名單（依傾向分數排序，取前 40）")
    top = ws.sort_values("score", ascending=False).head(40).copy()
    short = ["存", "卡", "貸", "財", "保"]
    top["現有商品"] = top.apply(
        lambda r: "".join(short[i] for i in range(5) if r[FLAGS[i]] == 1), axis=1)
    top["AUM(萬)"] = (top["TotalAUM_k"] / 10).round().astype(int)
    top["建議商品"] = t_name
    show = top[["CustomerID", "Segment", "Age", "TenureYears", "AUM(萬)",
                "現有商品", "建議商品", "score"]].rename(
        columns={"CustomerID": "客戶ID", "Segment": "客群", "Age": "年齡",
                 "TenureYears": "往來年資", "score": "傾向分數"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)
