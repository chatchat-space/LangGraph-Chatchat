from typing import Type, Annotated, Optional, TypedDict
from abc import abstractmethod

import streamlit as st
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, filter_messages
from langgraph.graph.state import CompiledStateGraph

from chatchat.server.utils import build_logger

logger = build_logger()

__all__ = [
    "Graph",
    "State",
    "register_graph",
    "list_graph_titles_by_label",
    "get_graph_class_by_label_and_title",
    "get_graph_class"
]


class State(TypedDict):
    """
    定义一个基础 State 供 各类 graph 继承, 其中:
    1. messages 为所有 graph 的核心信息队列, 所有聊天工作流均应该将关键信息补充到此队列中;
    2. history 为所有工作流单次启动时获取 history_len 的 messages 所用(节约成本, 及防止单轮对话 tokens 占用长度达到 llm 支持上限),
    history 中的信息理应是可以被丢弃的.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    history: Optional[list[BaseMessage]]


# 全局字典用于存储不同类型图的名称和对应的类
rag_registry = {}
agent_registry = {}
graph_registry = {}


def register_graph(cls):
    # 将类注册到相应的注册表中
    label = cls.label
    name = cls.name
    title = cls.title

    if label == "rag":
        rag_registry[name] = {
            "class": cls,
            "title": title
        }
    elif label == "agent":
        agent_registry[name] = {
            "class": cls,
            "title": title
        }
    else:
        raise ValueError(f"Unknown label '{label}' for class '{cls.__name__}'.")

    graph_registry[name] = {
        "class": cls,
        "title": title
    }

    return cls


class Graph:
    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver,
                 *args,
                 **kwargs):
        self.llm = llm
        self.tools = tools
        self.history_len = history_len
        self.checkpoint = checkpoint

    @abstractmethod
    def get_graph(self) -> CompiledStateGraph:
        """
        定义了 graph 流程, 子类必须实现.
        """
        pass

    @abstractmethod
    def handle_event(self, *args, **kwargs):
        """
        定义了 graph 的消息返回处理逻辑, 子类必须实现.
        """
        pass

    async def async_history_manager(self, state: Type[State]) -> Type[State]:
        """
        目的: 节约成本.
        做法: 给 llm 传递历史上下文时, 把 AIMessage(Function Call) 和 ToolMessage 过滤, 只保留 history_len 长度的 AIMessage
        和 HumanMessage 作为历史上下文.
        todo: 目前 history_len 直接截取了 messages 长度, 希望通过 对话轮数 来限制.
        todo: 原因: 一轮对话会追加数个 message, 但是目前没有从 snapshot(graph.get_state) 中找到很好的办法来获取一轮对话.
        """
        try:
            filtered_messages = []
            for message in filter_messages(state["messages"], exclude_types=[ToolMessage]):
                if isinstance(message, AIMessage) and message.tool_calls:
                    continue
                filtered_messages.append(message)
            state["history"] = filtered_messages[-self.history_len:]
            return state
        except Exception as e:
            raise Exception(f"Filtering messages error: {e}")

    @staticmethod
    async def break_point(state: Type[State]) -> Type[State]:
        """
        用来在 graph 中增加断点, 暂停 graph.
        """
        print("---BREAK POINT---")
        return state

    @staticmethod
    async def human_feedback(state: Type[State]) -> Type[State]:
        """
        获取用户反馈后的处理.
        例如，等待用户输入并更新 state["user_feedback"]
        """
        print("---HUMAN FEEDBACK---")
        return state

    @staticmethod
    async def init_docs(state: Type[State]) -> Type[State]:
        """
        在知识库检索后, 将检索出来的知识文档提取出来.
        """
        state["docs"] = state["messages"][-1].content
        # ToolMessage 默认不会往 history 队列中追加消息, 需要手动追加
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])
        return state


@st.cache_data
def list_graph_titles_by_label(label: str) -> list[str]:
    if label == "rag":
        return [info["title"] for info in rag_registry.values()]
    elif label == "agent":
        return [info["title"] for info in agent_registry.values()]
    else:
        raise ValueError(f"Unknown label '{label}'.")


@st.cache_data
def get_graph_class_by_label_and_title(label: str, title: str) -> Type[Graph]:
    if label == "rag":
        for info in rag_registry.values():
            if info["title"] == title:
                return info["class"]
    elif label == "agent":
        for info in agent_registry.values():
            if info["title"] == title:
                return info["class"]
    else:
        raise ValueError(f"Unknown label '{label}'.")
    raise ValueError(f"No graph found with title '{title}' for label '{label}'.")


def get_graph_class(name: str) -> Type[Graph]:
    if name not in graph_registry:
        raise ValueError(f"Graph '{name}' is not registered in graph_registry.")
    return graph_registry[name]["class"]
