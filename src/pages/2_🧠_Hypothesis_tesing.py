# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# from scipy.special import comb

st.set_page_config(
    page_title="假设检验",
    page_icon="🧠",
)

st.title("假设检验")

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

C1, C2, C3 = st.tabs(["为什么需要", "怎么做", "补充细节"])

if st.button("再来一次"):
    pass

with C1:
    st.subheader("回到抛硬币")
    "我们现在手头有一个硬币，但是不确定是否是均匀的，它可能是偏向正面的，也可能是偏向反面的。我们想要知道这个硬币是否是均匀的，也就是说，正反面出现的概率是否相等。"
    "我们可以通过抛硬币来验证这个假设，假设硬币是均匀的，那么正反面出现的概率应该是相等的。"
    N = int(st.number_input("抛硬币次数", 1, 100, 10))
    with st.form("一个小实验", clear_on_submit=True):
        st.write("一个小实验")
        p = np.random.choice([0.4, 0.5, 0.6])
        samples = np.random.choice([0, 1], (N,), p=[1 - p, p])
        st.bar_chart(
            pd.Series(
                {"正面": samples.sum(), "反面": N - samples.sum()}, name=f"抛{N:d}次硬币的结果"
            )
        )
        guess = st.select_slider("你认为这枚硬币是？", ["偏反面的", "均匀的", "偏正面的"], value="均匀的")
        if st.form_submit_button("确认"):
            if (
                (guess == "均匀的" and p > 0.45 and p < 0.55)
                or (guess == "偏反面的" and p < 0.45)
                or (guess == "偏正面的" and p > 0.55)
            ):
                st.success("你猜对了！")
            else:
                st.error("你猜错了！")
    st.subheader("制定标准！")
    "不知道你大约需要抛几次才能达到比较高的正确率，也不知道你是如何猜的。不过一种简单的策略是，如果正反面次数相差过大，我们就认为硬币是不均匀的。根据这个策略，我们可以自动化得猜测硬币是否均匀。"
    tolerence = st.number_input(
        "允许最大正反面差值", 0, N, 1, help="比如当允许最大差值为 1 时，如果结果是 6:4，我们就认为硬币是不均匀的。"
    )
    M = int(st.number_input("最多实验次数", 1, 1000, 100))
    ps = np.random.choice([0.4, 0.5, 0.6], M)
    samples = (np.random.rand(M, N) < ps[:, None]).astype(int)
    pos = samples.sum(axis=1)
    neg = N - pos
    guesses = np.where(np.abs(pos - neg) <= tolerence, 0, np.where(pos > neg, 1, -1))
    random_guesses = np.random.choice([-1, 0, 1], M)
    results = guesses == np.where(ps == 0.5, 0, np.where(ps > 0.5, 1, -1))
    step_result = np.cumsum(results)
    random_results = random_guesses == np.where(ps == 0.5, 0, np.where(ps > 0.5, 1, -1))
    step_random_result = np.sum(random_results)
    st.bar_chart(
        pd.Series({"你的策略": np.sum(results), "随机猜测": np.sum(random_results)}, name="准确率")
    )
    f"尝试几次就不难发现，这个策略根据允许的最大差值不同，效果差异也比较明显，并且也并不是单调变化的，但是要一遍一遍尝试合适的值还是不太方便，更何况这只是模拟实验，还不是真的要你抛{M*N:d}次硬币并且计数呢！那么有没有更聪明的策略呢？"
    st.subheader("换个思路？")
    "答案显然是有的，俗话说得好，事出反常必有妖。比如我们假设硬币是均匀的，那么就可以计算出这样的一个结果（乃至更极端的结果）出现的概率是多少，比如我们抛 10 次，其中 8 次是正面，那么就有："
    st.latex(
        r"\begin{aligned}P(X\geq 8) = &P(X=8)+P(X=9)+P(X=10)\\ = & \frac{C_{10}^{8}}{2^{10}} + \frac{C_{10}^{9}}{2^{10}} + \frac{C_{10}^{10}}{2^{10}}\\ \approx & 0.044+0.010+0.001\\ = & 0.055 \end{aligned}"
    )
    "既然我们已经可以计算出每次事件发生的概率，那么我们就可以预先设定好一个阈值，低于这个阈值的概率我们就认为是反常的，也就根据这个来判断硬币是不是均匀的了"
    P = st.slider("阈值", 0.0, 1.0, 0.05, 0.01)
    p_vals = np.zeros_like(pos)
    for i, p in enumerate(pos):
        p_vals[i] = stats.binom_test(p, N)
    P_guess = np.where(p_vals > P, 0, np.where(pos > neg, 1, -1))
    P_result = P_guess == np.where(ps == 0.5, 0, np.where(ps > 0.5, 1, -1))
    P_step_result = np.cumsum(P_result)
    st.bar_chart(
        pd.Series(
            {
                "原始策略": np.sum(results),
                "随机猜测": np.sum(random_results),
                "新的策略": np.sum(P_result),
            },
            name="准确率",
        )
    )
    "看！新的策略和之前的一样，都能甩开乱猜的策略一大截，并且这套拿发生概率来决策的框架也比之前的策略更容易推广了！"

with C2:
    st.subheader("假设检验四步走")

    with st.expander("1. 研究问题"):
        "在各类实验中我们通常比较关心均值，因此我们这里也主要以均值检验为例。我们需要一个零假设 $H_0$ 和一个备选假设 $H_\\alpha$ 来描述我们的问题"
        options = ["A≤B", "A=B", "A≥B"]
        neg_options = ["A>B", "A≠B", "A<B"]
        test_choice = options.index(
            st.select_slider("想验证的 A 和 B 人群平均寿命的关系", options, options[1])
        )
        st.latex(
            r"\begin{cases}H_0:A\text{人群期望寿命}"
            + (r"=" if test_choice == 1 else r"\ge " if test_choice == 2 else r"\le ")
            + r"B\text{人群期望寿命}\\"
            + r"H_\alpha:A\text{人群期望寿命}"
            + (r"\ne " if test_choice == 1 else r"< " if test_choice == 2 else r"> ")
            + r"B\text{人群期望寿命}\\"
            + r"\end{cases}"
        )

    with st.expander("2. 收集数据"):
        "顾名思义，就是去收集对应的数据啦，好比前面重复抛硬币的实验"
        L, R = st.columns(2)
        with L:
            A = int(st.number_input("A 人群样本数", 1, 100, 10))
        with R:
            B = int(st.number_input("B 人群样本数", 1, 100, 10))
        A_mean, B_mean = np.random.randint(70, 75, 2)
        A_std, B_std = np.random.randint(1, 10, 2)
        A_dist = stats.norm(loc=A_mean, scale=A_std)
        B_dist = stats.norm(loc=B_mean, scale=B_std)
        A_samples = A_dist.rvs(size=A)
        B_samples = B_dist.rvs(size=B)
        A_kernel = stats.gaussian_kde(A_samples)
        B_kernel = stats.gaussian_kde(B_samples)
        X_plot = np.linspace(
            min(A_dist.ppf(0.001), B_dist.ppf(0.001)),
            max(A_dist.ppf(0.999), B_dist.ppf(0.999)),
            1000,
        )
        st.area_chart(
            pd.DataFrame(
                {"A 人群样本分布": A_kernel(X_plot), "B 人群样本分布": B_kernel(X_plot)},
                index=X_plot,
            )
        )

    with st.expander("3. 统计检验"):
        "我们需要一个统计量来衡量我们的数据和假设的差异，这个统计量的分布应该服从一个已知的分布，这样我们就可以用这个分布的累积分布函数来计算我们的检验统计量的概率，从而得到我们的 p 值。"
        "这一步通常交给计算机来完成，我们只需要选择合适的检验方式即可，比如选择 t 检验："
        lang = st.radio("选择语言", ["Python", "R 语言"])
        if lang == "Python":
            st.code(
                f"""
from scipy import stats

stats.ttest_ind(
    A,                  # A 人群样本
    B,                  # B 人群样本
    equal_var=False,    # 假设方差不等
    alternative="{(
        "two-sided"
        if test_choice == 1
        else "less"
        if test_choice == 2
        else "greater"
    )}", # 备选假设 {neg_options[test_choice]}
).pvalue                # 取结果中这个神秘的 P 值
"""
            )
        else:
            "- R 语言"
            st.code(
                f"""
t.test(
    A,                  # A 人群样本
    B,                  # B 人群样本
    var.equal = FALSE,  # 假设方差不等，默认
    alternative="{(
        "two.sided"
        if test_choice == 1
        else "less"
        if test_choice == 2
        else "greater"
    )}", # 备选假设 {neg_options[test_choice]}
)$p.value               # 取结果中这个神秘的 P 值
""",
                language="r",
            )
        p_val = stats.ttest_ind(
            A_samples,
            B_samples,
            equal_var=False,
            alternative=(
                "two-sided"
                if test_choice == 1
                else "less"
                if test_choice == 2
                else "greater"
            ),
        ).pvalue
        "然后我们就能得到这个 P 值了，这也是浓缩了我们检验结果的指标："
        st.latex(r"P\text{值} = " f"{p_val:.4f}")

    with st.expander("4. 得出结论"):
        "如同前面抛硬币的实验一样，我们现在可以通过衡量这个 P 值的大小来选择接受零假设还是拒绝零假设了，这个特定的阈值就是我们的显著性水平（通常为 0.01, 0.05, 0.1 等）："
        alpha = st.slider("显著性水平 ɑ", 0.01, 0.1, 0.05, 0.01)
        "如果 P 值小于显著性水平，我们就拒绝零假设，否则我们就接受零假设。"
        if p_val < alpha:
            "我们拒绝零假设，接受备选假设（强证明）："
            st.latex(r"H_1: " f"{neg_options[test_choice]}")
        else:
            "我们没有证据推翻零假设（弱证明）："
            st.latex(r"H_0: " f"{options[test_choice]}")

with C3:
    st.subheader("P 值到底是什么")
    "依照定义来说，P 值是一个概率值，代表了在**假定零假设成立**的情况下，我们得到这样结果数据的概率。"
    "通过设定显著性水平来控制假设检验的结果，实际上也是控制了我们由于小概率事件导致误判的机会。"
    st.subheader("第一类错误和第二类错误")
    "结合上面 P 值的定义，我们不难发现，控制 P 值大小只是针对了一种情况的错误，还有另外一种错误没有被考虑到，下面这个表区分了两种错误的类型。"
    st.table(
        pd.DataFrame(
            {"零假设成立": ["✔️", "❌ 第一类错误"], "备选假设成立": ["❌ 第二类错误", "✔️"]},
            index=["接受零假设", "拒绝零假设"],
        )
    )
    "这也是我们无能为力的地方，因为我们无法同时控制两种错误的概率，因此我们需要在这两种错误之间做出权衡。通常我们选择控制第一类错误的概率，比如在药效测试中，我们宁可把有效果的药误判为没有效果，也不能把没效果的药误判为有效果。"
    "直观上来看，我们是通过控制截断水平，将两个相近的分布区分开来，我们只是选择确保第一类错误的概率不会太大，而第二类错误的概率则是不确定的。"
    H0_dist = stats.norm(loc=0, scale=1)
    H1_dist = stats.norm(loc=3.5, scale=1)
    alpha = 2.0
    X_plot = np.linspace(
        min(H0_dist.ppf(0.001), H1_dist.ppf(0.001)),
        max(H0_dist.ppf(0.999), H1_dist.ppf(0.999)),
        1000,
    )

    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Scatter(
            x=X_plot,
            y=H0_dist.pdf(X_plot),
            mode="lines",
            name="零假设分布",
            # fill="tozeroy"
            line=dict(color="deepskyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_plot[X_plot > alpha],
            y=H0_dist.pdf(X_plot[X_plot > alpha]),
            mode="lines",
            name=f"第一类错误",
            fill="tozeroy",
            line=dict(color="deepskyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_plot,
            y=H1_dist.pdf(X_plot),
            mode="lines",
            name=f"备选假设分布",
            # fill="tozeroy",
            line=dict(color="mediumspringgreen"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X_plot[X_plot < alpha],
            y=H1_dist.pdf(X_plot[X_plot < alpha]),
            mode="lines",
            name=f"第二类错误",
            fill="tozeroy",
            line=dict(color="mediumspringgreen"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[alpha] * 2,
            y=[0, max(H0_dist.pdf(alpha), H1_dist.pdf(alpha))],
            mode="lines",
            line=dict(color="deeppink", width=3, dash="dash"),
            name="显著性水平",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("P 值小于显著性水平意味着什么")
    "P 值小于显著性水平，意味着我们拒绝了零假设，接受了备选假设，这叫做**统计上显著**。"
    "统计上显著并不意味着有差异巨大，只是代表“我们有足够的证据来证明差异存在”；反之，统计上不显著（P 值大）的情况下我们也不能断定差异不存在。"

    st.subheader("拓展阅读")
    "更多有关假设检验的知识，可以参考[此链接](https://www.scribbr.com/statistics/statistical-power)"