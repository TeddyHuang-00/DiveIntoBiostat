import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.optimize import curve_fit

os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="获取数据",
    page_icon="📊",
)

st.title("获取数据")

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

C1, C2, C3 = st.tabs(["酶活测定", "数学模型", "分析处理"])

if st.button("再来一次"):
    pass

with C1:
    "以测定 GST 酶活为例，我们将介绍如何获取数据并进行后续分析："
    "在实验中，GST 酶会催化 GSH 与 CDNB 结合生成 GS-DNB，产物在 340nm 波长处有一个吸收峰，因此我们可以通过测定溶液的吸光度来测定产物的生成量，进而间接测定 GST 的酶活。"
    st.subheader("分光光度计原理")
    "测量吸光度的部分就需要用到分光光度计，其原理如下图所示："

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=[0, 0, 1, 1, 0],
            y=[0, 4, 4, 0, 0],
            name="比色皿",
            mode="lines",
            line=dict(color="lightslategrey"),
            fill="toself",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0, 1, 1, 0],
            y=[0, 3, 3, 0, 0],
            name="样本溶液",
            mode="lines",
            line=dict(color="thistle"),
            fill="toself",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(-2, 0, 100),
            y=[1.5] * 100,
            name="入射光路",
            mode="lines",
            line=dict(color="white"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, -1, 100),
            y=np.linspace(1.5, 2, 100),
            name="反射光路",
            mode="lines",
            line=dict(color="grey", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 0.75, 100),
            y=np.linspace(1.5, 2.25, 100),
            name="散射光路",
            mode="lines",
            line=dict(color="silver", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=[1.5] * 100,
            name="吸收光路",
            mode="lines",
            line=dict(color="whitesmoke", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(1, 3, 100),
            y=[1.5] * 100,
            name="透射光路",
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
                # "showgrid": False,
                "showticklabels": False,
                "visible": False,
            },
            "yaxis": {
                # "showgrid": False,
                "showticklabels": False,
                "visible": False,
            },
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    "我们将入射光强记为 $I$，反射光强记为 $R$，散射光强记为 $S$，吸收光强记为 $A$，透射光强记为 $I_0$，则有："
    st.latex(r"I=R+S+A+I_0")
    "如果用不含有样品的空白体系进行测定，则此时不包含吸收光强，即只有："
    st.latex(r"I=R+S+I_0")
    "因此使用空白体系作为“空白”来补偿入射光的损失，则可以得到简化的公式："
    st.latex(r"\tilde{I}=A+I_0")
    "由此得到我们的吸光度公式："
    st.latex(r"Abs=-\log\frac{I}{I_0}")
    "结合透射率的定义 $T=\\frac{I}{I_0}$ 和比尔-朗伯定律 $T=e^{-\\varepsilon Lc}$ 可知："
    st.latex(r"Abs=-\log T=\varepsilon Lc")
    "其中 $\\varepsilon$ 为摩尔消光系数，$L$ 为光程长度，$c$ 为溶质摩尔浓度。"
    "基于此，我们可以代入 GS-DNB 的摩尔消光系数与测量中的光程长度，轻松得到溶质浓度。"

with C2:
    st.subheader("原始数据处理")
    "好吧，我们现在已经做完一组实验了，我们每隔一段时间记录一次吸光度，就可以得到如图所示的曲线了。"

    def model(t, K, S=1.0, epsilon=9.6, L=1.0):
        return (1 - np.exp(-K * t)) * S * epsilon * L

    t = np.arange(0, 4.5, 0.5)
    K = np.random.uniform(0.1, 0.3)
    S = np.random.uniform(0.8, 1.0)
    epislon = 9.6
    L = 1.0
    Abs = model(t, K, S, epislon, L) * (1 + np.random.normal(0, 0.025, len(t)))
    st.line_chart(pd.DataFrame({"吸光度曲线": Abs}, index=t))
    "结合前面提到的吸光度和物质浓度的关系，我们可以得到如下的产物浓度曲线："
    st.latex(r"c=\frac{Abs}{\varepsilon L}")
    st.line_chart(pd.DataFrame({"产物浓度曲线": Abs / (epislon * L)}, index=t))
    "我们有了产物浓度曲线，但是我们想要测定的是样品中的酶活，因此我们还需要讲产物浓度转换为酶活。这里就需要涉及到一些有关反应的建模了。"
    "根据[1]中说明，GSH 与 CDNB 的反应本身为基元反应，即只有一个过渡态，见下图。"
    st.image("../../assets/GSH-CDNB-glow.png")
    "因此我们可以使用简单的动力学来描述类似的酶催化过程过程，即："
    st.latex(
        r"E+S \xrightleftharpoons[k_{-1}]{k_1} ES \xrightleftharpoons[k_{-2}]{k_2} E+P"
    )
    "其中"
    st.latex(
        r"\begin{aligned}E: & \text{酶}\\ S: & \text{底物}\\ P: & \text{产物}\end{aligned}"
    )
    "在 GST 酶的催化下，应当有过渡态到产物的生成过程基本不可逆，即有："
    st.latex(r"E+S \xrightleftharpoons[k_{-1}]{k_1} ES \xrightharpoonup{k_2} E+P")
    "在较大的时间尺度下，反应体系应当处于准平衡态，因此有 $[ES]$ 浓度基本恒定，即消耗与生成速率相当："
    st.latex(r"(k_{-1}+k_2)[ES]=k_1[E][S]")
    "同时由于第二步反应较快，因此 $[ES]$ 应当只占总酶量的一小部分，即有"
    st.latex(r"[E] = E_0-[ES] = \left(1-\frac{[ES]}{E_0}\right) E_0 \approx E_0")
    "$E_0$ 为初始加入的酶总量，为常数，由此我们可以得到："
    st.latex(
        r"(k_{-1}+k_2)[ES]=k_1E_0[S]\\\Rightarrow [ES]=\frac{k_1E_0}{k_{-1}+k_2}[S]\\\Rightarrow\frac{d[P]}{dt}=k_2[ES]=\frac{k_1k_2E_0}{k_{-1}+k_2}[S]"
    )
    "由于底物与产物的化学计量数之比为 1，因此则应当有"
    st.latex(r"S_0 = [S] + [P] + [ES] \approx [S] + [P]")
    "因此我们可以得到："
    st.latex(
        r"\frac{d[P]}{dt}=\frac{k_1k_2E_0}{k_{-1}+k_2}\left(S_0-[P]\right)\\ \Rightarrow \frac{k_{-1}+k_2}{k_1k_2E_0}\frac{d[P]}{S_0-[P]}=dt"
    )
    "经由两边积分与整理可"
    st.latex(r"[P]=\left(1-\exp\left\{-\frac{k_1k_2E_0}{k_{-1}+k_2}t\right\}\right)S_0")
    "如果将 $\\frac{k_1k_2E_0}{k_{-1}+k_2}$ 设为总酶活 $K$，那么可以将上式简化为"
    st.latex(r"[P]=\left(1-e^{-Kt}\right)S_0")
    st.markdown(
        "这样我们就能结合试验数据，拟合一条曲线<sup>[2]</sup>，来得到我们的酶活了！真不错！", unsafe_allow_html=True
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
                "产物浓度": Abs / epislon / L,
                "拟合结果": model(t, popt[0], popt[1], epislon, L) / epislon / L,
            }
        )
    )
    st.latex(r"[P]=\left(1-e^{-" f"{popt[0]:.4f}" r"t}\right)\times" f"{popt[1]:.4f}")
    "当然这边的 $K$ 代表的是体系中的总酶量，与我们加入的酶量 $E_0$ 有关，因此我们也可以用 $K/E_0$ 来表示单位酶活"

    "---"
    st.caption(
        "1. Habdous M, Vincent-Viry M, Visvikis S, Siest G. Rapid spectrophotometric method for serum glutathione S-transferases activity. Clin Chim Acta. 2002 Dec;326(1-2):131-42. doi: 10.1016/s0009-8981(02)00329-7. PMID: 12417104."
    )
    st.caption(
        "2. 这里我们将 $S_0$ 也设为自由量，因为体系中实际能发生反应底物浓度可能与加入的底物浓度不尽相同，因此我们将其设为自由量以获得更好的拟合结果"
    )

with C3:
    st.subheader("汇总数据")
    "通过前面的这些操作，我们已经能够通过实验数据计算得到酶活了，最后就是联系我们前面所提的假设检验来验证我们的猜想了。"
    "比方说我们在不同 pH 值的情况下测定了酶活，得到了如下的数据："
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
        yaxis_title="归一化后酶活",
        showlegend=False,
    )
    st.plotly_chart(fig)

    st.subheader("假设与显著性水平")
    "有了汇总的数据之后，我们可以再次明确我们的假设。这里我们想要研究的是 pH 值对酶活是否有影响，因此我们的假设可以设定为："
    st.latex(
        r"\begin{cases}H_0:&\text{所有 pH 下酶活均相等}\\H_\alpha:&\text{至少有一种 pH 下酶活不同于其他}\end{cases}"
    )
    "像这种多个组别之间的假设检验就没法使用 t-检验了，不过我们可以使用方差分析（Analysis of Variance, ANOVA）来完成这一工作，这一种检验方式的具体细节可以参考[1]。"
    "至于显著性水平，由于这边的样本量正合适，因此我们可以使用 0.05 作为显著性水平。当然你也可以选择一个自己喜欢的显著性水平。"
    alpha = st.select_slider("显著性水平", options=[0.01, 0.05, 0.1], value=0.05)

    st.subheader("写点代码")
    "顺带一提，在这类数据分析中，我们通常会遇到以下两种数据形式，由于纵表允许嵌套分组，并且可拓展性更好（任意添加字段或选用不同字段进行分析），因此我们一般会选择纵表来进行分析。"
    L, R = st.columns(2, gap="medium")
    with L:
        with st.expander("纵表 ✔️"):
            "列头为各个字段，每一行为一个样本，比如："
            st.dataframe(
                pd.DataFrame({"activity": Ks.flatten(), "pH": np.repeat(pH, 10)})
            )
    with R:
        with st.expander("横表 ❌"):
            "每一列为一组实验条件，每一格数据都是一个样本，比如："
            st.dataframe(pd.DataFrame(Ks, columns=pH))

    "好了，我们来写点代码来完成这一步骤吧！"
    lang = st.radio("选择语言", ["Python", "R 语言"], horizontal=True)
    if lang == "Python":
        st.code(
            f"""
from scipy import stats

stats.f_oneway(
    *data.groupby("pH")["activity"] # 将数据按照 pH 分组，
         .apply(list),              # 然后将酶活转换为列表
).pvalue                            # 取结果中的 P 值
"""
        )
    else:
        "- R 语言"
        st.code(
            f"""
aov(
    activity ~ pH, # 公式，表示研究酶活性与 pH 之间的关系
    data,          # 数据来源（必须包含公式中的字段）
)$p.value          # 取结果中的 P 值
""",
            language="r",
        )
    p_val = stats.f_oneway(*Ks.T).pvalue
    "然后我们就可以得到需要的 P 值了！"
    st.metric(
        "P 值",
        f"{p_val:.3e}",
        f"{'不' if p_val >= alpha else ''}显著",
        delta_color=("inverse" if p_val >= alpha else "normal"),
    )
    "这里结果是显然是统计上显著的，这意味着至少有一个 pH 条件下酶活的水平有显著区别。那么如果想要进一步细分具体是哪个 pH 值组导致结果的显著性，我们可以将其排除后再进行一次方差分析，这样就可以更进一步细化我们的结果了！"
    groups = st.multiselect("选择要排除的 pH 组", pH, default=pH[0], help="多于两个组的选择将默认只保留前两个")
    indices = [i for i, p in enumerate(pH) if p not in groups[:2]]
    p_val = stats.f_oneway(*Ks.T[indices]).pvalue
    st.metric(
        "P 值",
        f"{p_val:.3e}",
        f"{'不' if p_val >= alpha else ''}显著",
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
        yaxis_title="归一化后酶活",
        showlegend=False,
    )
    st.plotly_chart(fig)

    "---"
    st.caption(
        r"1. [Wikipedia: 方差分析](https://zh.wikipedia.org/wiki/%E6%96%B9%E5%B7%AE%E5%88%86%E6%9E%90)"
    )
