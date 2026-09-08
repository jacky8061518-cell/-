"""Trading desk entry point.

Navigation lives here so every page carries a real name in the sidebar and the
page configuration is applied exactly once per session. The pages themselves
only render; they do not configure the shell.
"""

from __future__ import annotations

import streamlit as st

from trading_desk.ui import configure_shell

configure_shell()

navigation = st.navigation(
    {
        "交易台": [
            st.Page("pages/0_總覽.py", title="總覽", icon="🛰️", default=True),
            st.Page("pages/1_訊號流.py", title="訊號流", icon="📡"),
            st.Page("pages/2_風險面板.py", title="風險面板", icon="🛡️"),
        ],
        "情報": [
            st.Page("pages/3_情報與敘事.py", title="情報與敘事", icon="🧭"),
            st.Page("pages/4_每日簡報.py", title="每日簡報", icon="📰"),
        ],
        "驗證與維運": [
            st.Page("pages/5_訊號成效.py", title="訊號成效", icon="🎯"),
            st.Page("pages/6_系統健康與回放.py", title="系統健康與回放", icon="🩺"),
            st.Page("pages/9_研究實驗室.py", title="研究實驗室", icon="🔬"),
        ],
    }
)

navigation.run()
