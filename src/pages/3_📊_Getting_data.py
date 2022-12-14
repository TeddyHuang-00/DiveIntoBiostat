import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.optimize import curve_fit

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="è·åæ°æ®",
    page_icon="ð",
)

st.title("è·åæ°æ®")

plotly_layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.5)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.25)"),
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    legend=dict(
        orientation="h",
        xanchor="center",
        yanchor="top",
        x=0.5,
        y=-0.05,
    ),
)

C1, C2, C3 = st.tabs(["é¶æ´»æµå®", "æ°å­¦æ¨¡å", "åæå¤ç"])

if st.button("åæ¥ä¸æ¬¡"):
    pass

with C1:
    "ä»¥æµå® GST é¶æ´»ä¸ºä¾ï¼æä»¬å°ä»ç»å¦ä½è·åæ°æ®å¹¶è¿è¡åç»­åæï¼"
    "å¨å®éªä¸­ï¼GST é¶ä¼å¬å GSH ä¸ CDNB ç»åçæ GS-DNBï¼äº§ç©å¨ 340nm æ³¢é¿å¤æä¸ä¸ªå¸æ¶å³°ï¼å æ­¤æä»¬å¯ä»¥éè¿æµå®æº¶æ¶²çå¸ååº¦æ¥æµå®äº§ç©ççæéï¼è¿èé´æ¥æµå® GST çé¶æ´»ã"
    st.subheader("ååååº¦è®¡åç")
    "æµéå¸ååº¦çé¨åå°±éè¦ç¨å°ååååº¦è®¡ï¼å¶åçå¦ä¸å¾æç¤ºï¼"

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=[0, 0, 1, 1, 0],
            y=[0, 4, 4, 0, 0],
            name="æ¯è²ç¿",
            mode="lines",
            line=dict(color="lightslategrey"),
            fill="toself",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0, 1, 1, 0],
            y=[0, 3, 3, 0, 0],
            name="æ ·æ¬æº¶æ¶²",
            mode="lines",
            line=dict(color="thistle"),
            fill="toself",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(-2, 0, 100),
            y=[1.5] * 100,
            name="å¥å°åè·¯",
            mode="lines",
            line=dict(color="white"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, -1, 100),
            y=np.linspace(1.5, 2, 100),
            name="åå°åè·¯",
            mode="lines",
            line=dict(color="grey", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 0.75, 100),
            y=np.linspace(1.5, 2.25, 100),
            name="æ£å°åè·¯",
            mode="lines",
            line=dict(color="silver", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=[1.5] * 100,
            name="å¸æ¶åè·¯",
            mode="lines",
            line=dict(color="whitesmoke", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(1, 3, 100),
            y=[1.5] * 100,
            name="éå°åè·¯",
            mode="lines",
            line=dict(color="grey"),
        )
    )
    fig.add_annotation(
        x=-1,
        y=1.5,
        ax=-1.5,
        ay=1.5,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=1,
        arrowcolor="white",
    )
    fig.add_annotation(
        x=2,
        y=1.5,
        ax=1.5,
        ay=1.5,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=1,
        arrowcolor="grey",
    )
    fig.update_layout(
        {
            "xaxis": {
                "showticklabels": False,
                "visible": False,
            },
            "yaxis": {
                "showticklabels": False,
                "visible": False,
            },
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    "æä»¬å°å¥å°åå¼ºè®°ä¸º $I$ï¼åå°åå¼ºè®°ä¸º $R$ï¼æ£å°åå¼ºè®°ä¸º $S$ï¼å¸æ¶åå¼ºè®°ä¸º $A$ï¼éå°åå¼ºè®°ä¸º $I_0$ï¼åæï¼"
    st.latex(r"I=R+S+A+I_0")
    "å¦æç¨ä¸å«ææ ·åçç©ºç½ä½ç³»è¿è¡æµå®ï¼åæ­¤æ¶ä¸åå«å¸æ¶åå¼ºï¼å³åªæï¼"
    st.latex(r"I=R+S+I_0")
    "å æ­¤ä½¿ç¨ç©ºç½ä½ç³»ä½ä¸ºâç©ºç½âæ¥è¡¥å¿å¥å°åçæå¤±ï¼åå¯ä»¥å¾å°ç®åçå¬å¼ï¼"
    st.latex(r"\tilde{I}=A+I_0")
    "ç±æ­¤å¾å°æä»¬çå¸ååº¦å¬å¼ï¼"
    st.latex(r"Abs=-\log\frac{I}{I_0}")
    "ç»åéå°ççå®ä¹ $T=\\frac{I}{I_0}$ åæ¯å°-æä¼¯å®å¾ $T=e^{-\\varepsilon Lc}$ å¯ç¥ï¼"
    st.latex(r"Abs=-\log T=\varepsilon Lc")
    "å¶ä¸­ $\\varepsilon$ ä¸ºæ©å°æ¶åç³»æ°ï¼$L$ ä¸ºåç¨é¿åº¦ï¼$c$ ä¸ºæº¶è´¨æ©å°æµåº¦ã"
    "åºäºæ­¤ï¼æä»¬å¯ä»¥ä»£å¥ GS-DNB çæ©å°æ¶åç³»æ°ä¸æµéä¸­çåç¨é¿åº¦ï¼è½»æ¾å¾å°æº¶è´¨æµåº¦ã"

with C2:
    st.subheader("åå§æ°æ®å¤ç")
    "å¥½å§ï¼æä»¬ç°å¨å·²ç»åå®ä¸ç»å®éªäºï¼æä»¬æ¯éä¸æ®µæ¶é´è®°å½ä¸æ¬¡å¸ååº¦ï¼å°±å¯ä»¥å¾å°å¦å¾æç¤ºçæ²çº¿äºã"

    def model(t, K, S=1.0, epsilon=9.6, L=1.0):
        return (1 - np.exp(-K * t)) * S * epsilon * L

    t = np.arange(0, 4.5, 0.5)
    K = np.random.uniform(0.1, 0.3)
    S = np.random.uniform(0.8, 1.0)
    epislon = 9.6
    L = 1.0
    Abs = model(t, K, S, epislon, L) * (1 + np.random.normal(0, 0.025, len(t)))
    st.line_chart(pd.DataFrame({"å¸ååº¦æ²çº¿": Abs}, index=t))
    "ç»ååé¢æå°çå¸ååº¦åç©è´¨æµåº¦çå³ç³»ï¼æä»¬å¯ä»¥å¾å°å¦ä¸çäº§ç©æµåº¦æ²çº¿ï¼"
    st.latex(r"c=\frac{Abs}{\varepsilon L}")
    st.line_chart(pd.DataFrame({"äº§ç©æµåº¦æ²çº¿": Abs / (epislon * L)}, index=t))
    "æä»¬æäºäº§ç©æµåº¦æ²çº¿ï¼ä½æ¯æä»¬æ³è¦æµå®çæ¯æ ·åä¸­çé¶æ´»ï¼å æ­¤æä»¬è¿éè¦è®²äº§ç©æµåº¦è½¬æ¢ä¸ºé¶æ´»ãè¿éå°±éè¦æ¶åå°ä¸äºæå³ååºçå»ºæ¨¡äºã"
    "æ ¹æ®[1]ä¸­è¯´æï¼GSH ä¸ CDNB çååºæ¬èº«ä¸ºåºåååºï¼å³åªæä¸ä¸ªè¿æ¸¡æï¼è§ä¸å¾ã"
    st.image("../../assets/GSH-CDNB-glow.png")
    "å æ­¤æä»¬å¯ä»¥ä½¿ç¨ç®åçå¨åå­¦æ¥æè¿°ç±»ä¼¼çé¶å¬åè¿ç¨è¿ç¨ï¼å³ï¼"
    st.latex(
        r"E+S \xrightleftharpoons[k_{-1}]{k_1} ES \xrightleftharpoons[k_{-2}]{k_2} E+P"
    )
    "å¶ä¸­"
    st.latex(
        r"\begin{aligned}E: & \text{é¶}\\ S: & \text{åºç©}\\ P: & \text{äº§ç©}\end{aligned}"
    )
    "å¨ GST é¶çå¬åä¸ï¼åºå½æè¿æ¸¡æå°äº§ç©ççæè¿ç¨åºæ¬ä¸å¯éï¼å³æï¼"
    st.latex(r"E+S \xrightleftharpoons[k_{-1}]{k_1} ES \xrightharpoonup{k_2} E+P")
    "å¨è¾å¤§çæ¶é´å°ºåº¦ä¸ï¼ååºä½ç³»åºå½å¤äºåå¹³è¡¡æï¼å æ­¤æ $[ES]$ æµåº¦åºæ¬æå®ï¼å³æ¶èä¸çæéçç¸å½ï¼"
    st.latex(r"(k_{-1}+k_2)[ES]=k_1[E][S]")
    "åæ¶ç±äºç¬¬äºæ­¥ååºè¾å¿«ï¼å æ­¤ $[ES]$ åºå½åªå æ»é¶éçä¸å°é¨åï¼å³æ"
    st.latex(r"[E] = E_0-[ES] = \left(1-\frac{[ES]}{E_0}\right) E_0 \approx E_0")
    "$E_0$ ä¸ºåå§å å¥çé¶æ»éï¼ä¸ºå¸¸æ°ï¼ç±æ­¤æä»¬å¯ä»¥å¾å°ï¼"
    st.latex(
        r"(k_{-1}+k_2)[ES]=k_1E_0[S]\\\Rightarrow [ES]=\frac{k_1E_0}{k_{-1}+k_2}[S]\\\Rightarrow\frac{d[P]}{dt}=k_2[ES]=\frac{k_1k_2E_0}{k_{-1}+k_2}[S]"
    )
    "ç±äºåºç©ä¸äº§ç©çåå­¦è®¡éæ°ä¹æ¯ä¸º 1ï¼å æ­¤ååºå½æ"
    st.latex(r"S_0 = [S] + [P] + [ES] \approx [S] + [P]")
    "å æ­¤æä»¬å¯ä»¥å¾å°ï¼"
    st.latex(
        r"\frac{d[P]}{dt}=\frac{k_1k_2E_0}{k_{-1}+k_2}\left(S_0-[P]\right)\\ \Rightarrow \frac{k_{-1}+k_2}{k_1k_2E_0}\frac{d[P]}{S_0-[P]}=dt"
    )
    "ç»ç±ä¸¤è¾¹ç§¯åä¸æ´çå¯"
    st.latex(r"[P]=\left(1-\exp\left\{-\frac{k_1k_2E_0}{k_{-1}+k_2}t\right\}\right)S_0")
    "å¦æå° $\\frac{k_1k_2E_0}{k_{-1}+k_2}$ è®¾ä¸ºæ»é¶æ´» $K$ï¼é£ä¹å¯ä»¥å°ä¸å¼ç®åä¸º"
    st.latex(r"[P]=\left(1-e^{-Kt}\right)S_0")
    st.markdown(
        "è¿æ ·æä»¬å°±è½ç»åè¯éªæ°æ®ï¼æåä¸æ¡æ²çº¿<sup>[2]</sup>ï¼æ¥å¾å°æä»¬çé¶æ´»äºï¼çä¸éï¼", unsafe_allow_html=True
    )
    popt, pcov = curve_fit(
        model,
        t,
        Abs,
        p0=[1e-3, 1e-3],
        bounds=(0, np.inf),
    )
    st.line_chart(
        pd.DataFrame(
            {
                "äº§ç©æµåº¦": Abs / epislon / L,
                "æåç»æ": model(t, popt[0], popt[1], epislon, L) / epislon / L,
            }
        )
    )
    st.latex(r"[P]=\left(1-e^{-" f"{popt[0]:.4f}" r"t}\right)\times" f"{popt[1]:.4f}")
    "å½ç¶è¿è¾¹ç $K$ ä»£è¡¨çæ¯ä½ç³»ä¸­çæ»é¶éï¼ä¸æä»¬å å¥çé¶é $E_0$ æå³ï¼å æ­¤æä»¬ä¹å¯ä»¥ç¨ $K/E_0$ æ¥è¡¨ç¤ºåä½é¶æ´»"

    "---"
    st.caption(
        "1. Habdous M, Vincent-Viry M, Visvikis S, Siest G. Rapid spectrophotometric method for serum glutathione S-transferases activity. Clin Chim Acta. 2002 Dec;326(1-2):131-42. doi: 10.1016/s0009-8981(02)00329-7. PMID: 12417104."
    )
    st.caption(
        "2. è¿éæä»¬å° $S_0$ ä¹è®¾ä¸ºèªç±éï¼å ä¸ºä½ç³»ä¸­å®éè½åçååºåºç©æµåº¦å¯è½ä¸å å¥çåºç©æµåº¦ä¸å°½ç¸åï¼å æ­¤æä»¬å°å¶è®¾ä¸ºèªç±éä»¥è·å¾æ´å¥½çæåç»æ"
    )

with C3:
    st.subheader("æ±æ»æ°æ®")
    "éè¿åé¢çè¿äºæä½ï¼æä»¬å·²ç»è½å¤éè¿å®éªæ°æ®è®¡ç®å¾å°é¶æ´»äºï¼æåå°±æ¯èç³»æä»¬åé¢ææçåè®¾æ£éªæ¥éªè¯æä»¬ççæ³äºã"
    "æ¯æ¹è¯´æä»¬å¨ä¸å pH å¼çæåµä¸æµå®äºé¶æ´»ï¼å¾å°äºå¦ä¸çæ°æ®ï¼"
    pH = np.arange(6, 8, 0.5)
    K_mean = stats.norm.pdf(pH, loc=7, scale=1)
    K_std = (K_mean / np.max(K_mean)) ** 2 / 10
    Ks = np.random.normal(K_mean, K_std, size=(10, len(pH))).clip(0, None)

    fig = go.Figure(layout=plotly_layout)
    for i in range(len(pH)):
        fig.add_trace(
            go.Box(
                y=Ks[:, i],
                name=f"pH={pH[i]:.1f}",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig.update_layout(
        yaxis_title="å½ä¸ååé¶æ´»",
        showlegend=False,
    )
    st.plotly_chart(fig)

    st.subheader("åè®¾ä¸æ¾èæ§æ°´å¹³")
    "æäºæ±æ»çæ°æ®ä¹åï¼æä»¬å¯ä»¥åæ¬¡æç¡®æä»¬çåè®¾ãè¿éæä»¬æ³è¦ç ç©¶çæ¯ pH å¼å¯¹é¶æ´»æ¯å¦æå½±åï¼å æ­¤æä»¬çåè®¾å¯ä»¥è®¾å®ä¸ºï¼"
    st.latex(
        r"\begin{cases}H_0:&\text{ææ pH ä¸é¶æ´»åç¸ç­}\\H_\alpha:&\text{è³å°æä¸ç§ pH ä¸é¶æ´»ä¸åäºå¶ä»}\end{cases}"
    )
    "åè¿ç§å¤ä¸ªç»å«ä¹é´çåè®¾æ£éªå°±æ²¡æ³ä½¿ç¨ t-æ£éªäºï¼ä¸è¿æä»¬å¯ä»¥ä½¿ç¨æ¹å·®åæï¼Analysis of Variance, ANOVAï¼æ¥å®æè¿ä¸å·¥ä½ï¼è¿ä¸ç§æ£éªæ¹å¼çå·ä½ç»èå¯ä»¥åè[1]ã"
    "è³äºæ¾èæ§æ°´å¹³ï¼ç±äºè¿è¾¹çæ ·æ¬éæ­£åéï¼å æ­¤æä»¬å¯ä»¥ä½¿ç¨ 0.05 ä½ä¸ºæ¾èæ§æ°´å¹³ãå½ç¶ä½ ä¹å¯ä»¥éæ©ä¸ä¸ªèªå·±åæ¬¢çæ¾èæ§æ°´å¹³ã"
    alpha = st.select_slider("æ¾èæ§æ°´å¹³", options=[0.01, 0.05, 0.1], value=0.05)

    st.subheader("åç¹ä»£ç ")
    "é¡ºå¸¦ä¸æï¼å¨è¿ç±»æ°æ®åæä¸­ï¼æä»¬éå¸¸ä¼éå°ä»¥ä¸ä¸¤ç§æ°æ®å½¢å¼ï¼ç±äºçºµè¡¨åè®¸åµå¥åç»ï¼å¹¶ä¸å¯æå±æ§æ´å¥½ï¼ä»»ææ·»å å­æ®µæéç¨ä¸åå­æ®µè¿è¡åæï¼ï¼å æ­¤æä»¬ä¸è¬ä¼éæ©çºµè¡¨æ¥è¿è¡åæã"
    L, R = st.columns(2, gap="medium")
    with L:
        with st.expander("çºµè¡¨ âï¸"):
            "åå¤´ä¸ºåä¸ªå­æ®µï¼æ¯ä¸è¡ä¸ºä¸ä¸ªæ ·æ¬ï¼æ¯å¦ï¼"
            st.dataframe(
                pd.DataFrame({"activity": Ks.flatten(), "pH": np.repeat(pH, 10)})
            )
    with R:
        with st.expander("æ¨ªè¡¨ â"):
            "æ¯ä¸åä¸ºä¸ç»å®éªæ¡ä»¶ï¼æ¯ä¸æ ¼æ°æ®é½æ¯ä¸ä¸ªæ ·æ¬ï¼æ¯å¦ï¼"
            st.dataframe(pd.DataFrame(Ks, columns=pH))

    "å¥½äºï¼æä»¬æ¥åç¹ä»£ç æ¥å®æè¿ä¸æ­¥éª¤å§ï¼"
    lang = st.radio("éæ©è¯­è¨", ["Python", "R è¯­è¨"], horizontal=True)
    if lang == "Python":
        st.code(
            f"""
from scipy import stats

stats.f_oneway(
    *data.groupby("pH")["activity"] # å°æ°æ®æç§ pH åç»ï¼
         .apply(list),              # ç¶åå°é¶æ´»è½¬æ¢ä¸ºåè¡¨
).pvalue                            # åç»æä¸­ç P å¼
"""
        )
    else:
        st.code(
            f"""
aov(
    activity ~ pH, # å¬å¼ï¼è¡¨ç¤ºç ç©¶é¶æ´»æ§ä¸ pH ä¹é´çå³ç³»
    data,          # æ°æ®æ¥æºï¼å¿é¡»åå«å¬å¼ä¸­çå­æ®µï¼
)$p.value          # åç»æä¸­ç P å¼
""",
            language="r",
        )
    p_val = stats.f_oneway(*Ks.T).pvalue
    "ç¶åæä»¬å°±å¯ä»¥å¾å°éè¦ç P å¼äºï¼"
    st.metric(
        "P å¼",
        f"{p_val:.3e}",
        f"{'ä¸' if p_val >= alpha else ''}æ¾è",
        delta_color=("inverse" if p_val >= alpha else "normal"),
    )
    "è¿éç»ææ¯æ¾ç¶æ¯ç»è®¡ä¸æ¾èçï¼è¿æå³çè³å°æä¸ä¸ª pH æ¡ä»¶ä¸é¶æ´»çæ°´å¹³ææ¾èåºå«ãé£ä¹å¦ææ³è¦è¿ä¸æ­¥ç»åå·ä½æ¯åªä¸ª pH å¼ç»å¯¼è´ç»æçæ¾èæ§ï¼æä»¬å¯ä»¥å°å¶æé¤ååè¿è¡ä¸æ¬¡æ¹å·®åæï¼è¿æ ·å°±å¯ä»¥æ´è¿ä¸æ­¥ç»åæä»¬çç»æäºï¼"
    groups = st.multiselect("éæ©è¦æé¤ç pH ç»", pH, default=pH[0], help="å¤äºä¸¤ä¸ªç»çéæ©å°é»è®¤åªä¿çåä¸¤ä¸ª")
    indices = [i for i, p in enumerate(pH) if p not in groups[:2]]
    p_val = stats.f_oneway(*Ks.T[indices]).pvalue
    st.metric(
        "P å¼",
        f"{p_val:.3e}",
        f"{'ä¸' if p_val >= alpha else ''}æ¾è",
        delta_color=("inverse" if p_val >= alpha else "normal"),
    )

    fig = go.Figure(layout=plotly_layout)
    for i in indices:
        fig.add_trace(
            go.Box(
                y=Ks[:, i],
                name=f"pH={pH[i]:.1f}",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    fig.update_layout(
        yaxis_title="å½ä¸ååé¶æ´»",
        showlegend=False,
    )
    st.plotly_chart(fig)

    "---"
    st.caption(
        r"1. [Wikipedia: æ¹å·®åæ](https://zh.wikipedia.org/wiki/%E6%96%B9%E5%B7%AE%E5%88%86%E6%9E%90)"
    )
