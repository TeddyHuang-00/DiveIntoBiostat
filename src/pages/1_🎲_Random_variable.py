# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="随机变量",
    page_icon="🎲",
)

st.title("随机变量")

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

C1, C2, C3 = st.tabs(
    [
        "抛硬币",
        "离散与连续",
        "中心极限定律",
    ]
)

if st.button("再来一次"):
    pass

with C1:
    "从抛硬币开始，假设我们有一枚均匀的硬币（正：🌝，反：🌚），那么抛十次的结果可能是如下这样："
    ten_tosses = np.random.randint(0, 2, 10)
    st.header(" ".join(["🌝" if i else "🌚" for i in ten_tosses]))
    "这样不太容易记录结果，所以我们引进一个定义："
    st.latex(r"X = \begin{cases} 1 & \text{正面} \\ 0 & \text{反面} \end{cases}")
    "那么我们抛十次的结果就可以用一个序列来表示："
    st.latex(r"X = \begin{bmatrix}" f"{'&'.join(map(str,ten_tosses))}" r"\end{bmatrix}")
    "看起来清楚多了，这样的符号记法中我们把 $X$ 称为随机变量，把 $X$ 的取值称为随机变量的取值。"
    "从名字上我们就知道，随机变量的具体取值是**不唯一、随机**的，可以点下面的按钮来看看再尝试一次的结果："

with C2:
    st.subheader("随机变量分类")
    "我们把随机变量的取值简单分为两类：离散型和连续型"
    discrete, continuous = st.columns(2, gap="medium")
    with discrete:
        "离散型随机变量的取值是**可数**的"
        with st.expander("例子"):
            "- 抛硬币：正、反"
            "- 掷骰子：1、2、3、4、5、6"
            "- 年级：大一、大二、大三、大四"
        "在离散型随机变量中，取值外的值是不存在的"
        with st.expander("例子"):
            "- 抛硬币：既正又反？"
            "- 掷骰子：3.5？"
            "- 年级：大一半？"
    with continuous:
        "连续型随机变量的取值是**不可数**的"
        with st.expander("例子"):
            "- 身高：1.6m、1.61m、1.62m……"
            "- 气温：10℃、10.1℃、10.2℃……"
            "- 风速：1m/s、1.01m/s、1.02m/s……"
        "而在连续型随机变量中，取值是可以无限细分的"
        with st.expander("例子"):
            "- 身高：1.81m、1.801m、1.8001m……"
            "- 气温：15.5℃、15.75℃、15.875℃……"
            "- 风速：1.5m/s、0.15m/s、0.015m/s……"
    st.subheader("概率分布")
    "对于一般的随机变量来说，我们需要用取值的概率来描述它，这样的概率称为**概率分布**"
    "在离散型随机变量中，这比较简单，我们只需要给出每个取值的概率即可，通常可以列成一个表格："
    st.table(pd.Series({"正面": 0.5, "反面": 0.5}, name="概率"))
    "更直观的，我们也可以用一个条形图来表示："
    st.bar_chart(pd.Series({i + 1: 1 / 6 for i in range(6)}, name="掷骰子"))
    st.subheader("概率密度函数")
    "在连续型随机变量中，事情就相对复杂了，因为取值之间并没有明确的界限可供区分；一种想法是，我们可以把一定范围内的值归为一类，然后给出每个类的概率："
    num = int(1e6)
    samples = np.random.normal(0, 1, num)
    bins: int = st.slider("分成几个区间：", 1, 100, 10)
    st.bar_chart(pd.Series(np.histogram(samples, bins=bins)[0] / num, name="概率"))
    "可以看到，随着区间的不断细分，每个区间内的概率越来越接近于0，这样的结果显然不利于我们的分析，因此我们需要一种更好的方法来表示连续型随机变量的概率分布"
    "这就是**概率密度函数**的由来，我们可以把它看作是一个函数，它的输入是取值，输出是取值的概率密度"
    "密度函数的妙处在于，我们现在可以通过积分来计算取值的概率了："
    st.latex(r"P(a \leq X \leq b) = \int_a^b f(x) dx")
    "比如还是拿一个标准正态分布函数来说，我们可以求任意区间内的概率"
    a, b = st.slider("a和b的取值：", -5.0, 5.0, (-1.0, 2.5), help="试试把a和b调到一起")
    st.latex(f"P({a} \\leq X \\leq {b}) = {stats.norm.cdf(b) - stats.norm.cdf(a):.4f}")
    x_plot = np.linspace(-5, 5, 1000)
    y_plot = stats.norm.pdf(x_plot)

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=x_plot[(x_plot >= a) & (x_plot <= b)],
            y=y_plot[(x_plot >= a) & (x_plot <= b)],
            mode="lines",
            name=f"S={stats.norm.cdf(b) - stats.norm.cdf(a):.4f}",
            fill="tozeroy",
            line=dict(color="deepskyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            name="概率密度函数",
            # fill="tozeroy"
            line=dict(color="deepskyblue"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    "如果你尝试了把a和b调到一起，你会发现，这时候的概率就是 0 了，也就意味着单点的概率为 0！听起来不是很妙，因为我们知道实际生活中这样的值是有可能出现的，那么我们要怎么来描述这样的值出现的概率呢？"
    "这就是我们需要对概率的描述进行更细致的定义，比如在离散的情况下我们可以说成是"
    st.latex(r"P(X=1)=...")
    "在连续的情况下我们可以说成是"
    st.latex(r"P(X \leq 1)=...")
    "更一般的，我们会将概率描述为**出现比此值更极端的取值**的概率，即"
    st.latex(
        r"P(x)=\min\left\{P(X\leq x),P(X\geq x)\right\}=\min\left\{P(X\leq x),1-P(X\leq x)\right\}"
    )
    "按照此定义，我们可以求出标准正态分布中任意取值的概率"
    x: float = st.slider(
        "x的取值：", -5.0, 5.0, 1.0, help="依照此处定义求得的是单尾概率，在对称的分布中也可以简单转化为双尾概率"
    )
    st.latex(
        f"P({x}) ="
        + (
            r"\int_{-\infty}^{" f"{x}" r"}{f(x)dx}"
            if x <= 0
            else r"\int_{" f"{x}" r"}^{+\infty}{f(x)dx}"
        )
        + f"= {min(stats.norm.cdf(x),1-stats.norm.cdf(x)):.4f}"
    )

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=x_plot[(x_plot <= x) if x <= 0 else (x_plot >= x)],
            y=y_plot[(x_plot <= x) if x <= 0 else (x_plot >= x)],
            mode="lines",
            name=f"S={min(stats.norm.cdf(x),1-stats.norm.cdf(x)):.4f}",
            fill="tozeroy",
            line=dict(color="deepskyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            name="概率密度函数",
            # fill="tozeroy"
            line=dict(color="deepskyblue"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with C3:
    st.subheader("常见分布")
    dist_type = st.selectbox("分布类型", ["正态分布", "均匀分布", "t-分布"])
    match dist_type:
        case "正态分布":
            st.write("正态分布是最常见的连续分布，其概率密度函数为")
            st.latex(
                r"f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}"
            )
            st.write("正态分布由两个参数所控制，分别是均值 μ 和标准差 σ")
            mu: float = st.slider("均值 μ", -5.0, 5.0, 0.0, 0.1)
            sigma: float = st.slider("方差 σ", 0.1, 3.0, 1.0, 0.1)
            dist = stats.norm(loc=mu, scale=sigma**2)
            x_plot = np.linspace(
                min(dist.ppf(0.0001), -10), max(dist.ppf(0.9999), 10), 1000
            )
            y_plot = dist.pdf(x_plot)
            st.area_chart(pd.Series(y_plot, name="正态分布概率密度", index=x_plot))
            st.write("可以发现 μ 控制分布的位置，而 σ 主要控制分布的宽度（形状）")

        case "均匀分布":
            st.write("均匀分布是最简单的连续分布，其概率密度函数为")
            st.latex(
                r"f(x)=\begin{cases}\frac{1}{b-a}&x\in[a,b]\\0&\text{otherwise}\end{cases}"
            )
            a, b = st.slider("a和b区间", -10.0, 10.0, (0.0, 1.0), 0.1)
            dist = stats.uniform(loc=a, scale=b - a)
            x_plot = np.linspace(-10, 10, 2500)
            y_plot = dist.pdf(x_plot)
            st.area_chart(pd.Series(y_plot, name="均匀分布概率密度", index=x_plot))
            st.write("均匀分布比较简单，两端范围既控制了概率密度分布的区间，也决定了密度的大小（面积保持为 1 不变）")

        case "t-分布":
            st.write("t-分布是统计学中常见的连续分布，通常和小样本的假设检验联系，其概率密度函数为")
            st.latex(
                r"f(x)=\frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\cdot\Gamma(\frac{\nu}{2})}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}"
            )
            st.write("这里我们所展示的为标准化后的t-分布，因此仅有一个可变参数自由度 ν 用于控制形状")
            nu: int = st.slider("自由度 ν", 3, 50, 3)
            dist = stats.t(df=nu)
            x_plot = np.linspace(-10, 10, 2500)
            y_plot = dist.pdf(x_plot)
            y_norm = stats.norm.pdf(x_plot)
            st.area_chart(
                pd.DataFrame(
                    {
                        "t-分布概率密度": y_plot,
                        "标准正态分布概率密度": y_norm,
                    },
                    index=x_plot,
                )
            )
            st.write(
                "在标准 t-分布中，自由度 ν 只控制了分布的形状，并且和标准正态分布放在一起对比时，我们不难发现，随着 ν 增大，其形状也逐渐趋向于标准正态分布"
            )

    st.subheader("由样本估计")
    "当然在现实中我们往往没法得到分布的具体形式，而是只能得到遵循该分布的一些样本，比如我们可以从上述分布中抽 $N$ 个样本"
    N = int(st.number_input("N=", 1, 10000, 100))
    "然后我们可以通过样本估计分布的参数，比如均值和方差"
    samples = dist.rvs(size=N)
    kernel = stats.gaussian_kde(samples)
    sample_plot = kernel(x_plot)
    st.area_chart(
        pd.DataFrame(
            {
                f"原{dist_type}": y_plot,
                "抽样结果": sample_plot,
            },
            index=x_plot,
        )
    )
    "当然不是所有的样本分布与原来接近，因此我们也可以比较样本均值和样本方差与原分布的均值和方差"
    st.metric("样本均值", samples.mean(), samples.mean() - dist.mean())
    st.metric("样本标准差", samples.var(), samples.std() - dist.std())
    "我们如果通过样本的均值、标准差和真实分布的均值、标准差还有一定差距，如果我们想要通过样本来估计分布的参数，那么一种想法就是可以再多次采样，通过重复来进行更准确的估计"
    N = int(st.number_input("单次抽样的样本个数 N", 1, 1000, 10))
    M = int(st.number_input("重复抽样的次数 M", 1, 1000, 10))
    samples = dist.rvs(size=(N, M))
    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    kernel = stats.gaussian_kde(means)

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=kernel(x_plot),
            mode="lines",
            name="概率密度函数",
            fill="tozeroy",
            line=dict(color="deepskyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[means.mean()] * 2,
            y=[0, kernel(means.mean())[0]],
            mode="lines",
            line=dict(color="deepskyblue", width=3, dash="dash"),
            name="样本均值的均值",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[dist.mean()] * 2,
            y=[0, kernel(x_plot).max()],
            mode="lines",
            line=dict(color="deeppink", width=2, dash="dash"),
            name="真实均值",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    "当然样本的标准差也有类似的估计"
    kernel = stats.gaussian_kde(stds)

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=kernel(x_plot),
            mode="lines",
            name="概率密度函数",
            fill="tozeroy",
            line=dict(color="deepskyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[stds.mean()] * 2,
            y=[0, kernel(stds.mean())[0]],
            mode="lines",
            line=dict(color="deepskyblue", width=3, dash="dash"),
            name="样本标准差的均值",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[dist.std()] * 2,
            y=[0, kernel(x_plot).max()],
            mode="lines",
            line=dict(color="deeppink", width=2, dash="dash"),
            name="真实标准差",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    "对于样本均值，比较有意思的一点是，随着抽样次数的增加，样本均值的分布趋向于一个固定的分布"
    kernel = stats.gaussian_kde(means)
    X = np.linspace(dist.mean() - 3 * dist.std(), dist.mean() + 3 * dist.std(), 2500)
    Y = stats.norm.pdf(X, loc=dist.mean(), scale=dist.std() / np.sqrt(N))
    st.area_chart(pd.DataFrame({"样本均值": kernel(X), "某个分布": Y}, index=X))
    "这个神奇的分布也是一个正态分布，即"
    st.latex(r"\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{N}\right)")
    "其中 $\\bar{X}$ 代表样本均值，而这个正态分布的均值与原分布相同，方差为原来的 $\\frac{1}{N}$"
    "这也就是中心极限定理"
