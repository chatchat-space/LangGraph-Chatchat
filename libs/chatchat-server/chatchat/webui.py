import asyncio
import sys

import streamlit as st
import streamlit_antd_components as sac

from chatchat import __version__
from chatchat.webui_pages.graph_rag.rag import graph_rag_page
from chatchat.webui_pages.knowledge_base.knowledge_base import knowledge_base_page
from chatchat.webui_pages.graph_agent.graph import graph_agent_page
from chatchat.webui_pages.utils import *

api = ApiRequest(base_url=api_address())


async def main():
    is_lite = "lite" in sys.argv  # TODO: remove lite mode
    # 设置默认头像

    st.set_page_config(
        "LangGraph-Chatchat",
        get_img_base64("chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/chatchat-space/Langgraph-Chatchat",
            "Report a bug": "https://github.com/chatchat-space/Langgraph-Chatchat/issues",
            "About": f"""欢迎使用 Langgraph-Chatchat WebUI {__version__}！""",
        },
        layout="centered",
    )

    # use the following code to set the app to wide mode and the html markdown to increase the sidebar width
    st.markdown(
        """
        <style>
        [data-testid="stSidebarUserContent"] {
            padding-top: 20px;
        }
        .block-container {
            padding-top: 25px;
        }
        [data-testid="stBottomBlockContainer"] {
            padding-bottom: 20px;
        }
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(
            get_img_base64("logo-long-langraph-chatchat.jpg"), use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：{__version__}</p>""",
            unsafe_allow_html=True,
        )

        selected_page = sac.menu(
            [
                sac.MenuItem("Agent 对话", icon="robot"),
                sac.MenuItem("RAG 对话", icon="book"),
                sac.MenuItem("知识库管理", icon="database"),
            ],
            key="selected_page",
            open_index=0,
        )

        sac.divider()

    if selected_page == "知识库管理":
        knowledge_base_page(api=api, is_lite=is_lite)
    elif selected_page == "RAG 对话":
        await graph_rag_page(api=api)
    elif selected_page == "Agent 对话":
        await graph_agent_page()


if __name__ == "__main__":
    asyncio.run(main())
