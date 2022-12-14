import streamlit as st

st.set_page_config(
    page_title="索引",
    page_icon="🗂️",
)

st.title("欢迎来到生物统计快速入门！")
st.subheader("↖️在侧边栏你能找到各个章节的链接")
st.subheader("建议设置为暗色模式以获得更好的阅读体验↗️")

with st.expander("Tips"):

    st.slider("在章节中存在一些可交互的小部件，比如你可以拖动他们👇", 0, 100, 50)

    st.selectbox("你也可以在这里选择一些选项", ["🌝", "🌚"])

    st.checkbox("或者勾选一些复选框", True)

    st.radio("或者选择一些单选框", ["🌝", "🌚"], horizontal=True)

    st.text_input("你也可以在这里输入一些文字", "点我编辑")

    st.line_chart(
        {"1. 图表大多是可交互的": list(range(9, -1, -1)), "2. 双击图表恢复默认": list(range(10))}
    )
