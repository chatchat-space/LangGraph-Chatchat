from typing import Callable, Any, Dict, Type, Annotated, List, Optional, TypedDict, TypeVar
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, filter_messages
from langgraph.graph.state import CompiledStateGraph

from chatchat.server.utils import build_logger

logger = build_logger()

__all__ = [
    "Graph",
    "regist_graph",
    "InputHandler",
    "EventHandler",
    "State",
    # "Response",
    "async_history_manager",
    "human_feedback",
    "break_point",
    "register_graph",
    "list_graph_titles_by_label",
    "get_graph_class_by_label_and_title"
]

_GRAPHS_REGISTRY: Dict[str, Dict[str, Any]] = {}


class State(TypedDict):
    """
    定义一个基础 State 供 各类 graph 继承, 其中:
    1. messages 为所有 graph 的核心信息队列, 所有聊天工作流均应该将关键信息补充到此队列中;
    2. history 为所有工作流单次启动时获取 history_len 的 messages 所用(节约成本, 及防止单轮对话 tokens 占用长度达到 llm 支持上限),
    history 中的信息理应是可以被丢弃的.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    history: Optional[list[BaseMessage]]


class Response(TypedDict):
    node: str
    content: Any


class Message(TypedDict):
    role: str
    content: str


class InputHandler(ABC):
    def __init__(self, query: str, metadata: Dict[str, Any]):
        self.query = query
        self.metadata = metadata  # 暂未使用

    def create_inputs(self) -> Dict[str, Any]:
        return {"messages": Message(role="user", content=self.query)}


class EventHandler(ABC):
    @abstractmethod
    def handle_event(self, node: str, events: Any) -> str:
        pass


# 定义一个类型变量，可以是各种 GraphState
T = TypeVar('T')


# 目的: 节约成本.
# 做法: 给 llm 传递历史上下文时, 把 AIMessage(Function Call) 和 ToolMessage 过滤, 只保留 history_len 长度的 AIMessage 作为历史上下文.
# todo: """目前 history_len 直接截取了 messages 长度, 希望通过 对话轮数 来限制.
#  原因: 一轮对话会追加数个 message, 但是目前没有从 snapshot(graph.get_state) 中找到很好的办法来获取一轮对话."""
async def async_history_manager(state: T, history_len: int, exclude_types: Optional[List[Type[BaseMessage]]] = None) \
        -> T:
    try:
        if exclude_types is None:
            exclude_types = [ToolMessage]
        filtered_messages = []
        for message in filter_messages(state["messages"], exclude_types=exclude_types):
            if isinstance(message, AIMessage) and message.tool_calls:
                continue
            filtered_messages.append(message)
        state["history"] = filtered_messages[-history_len:]
        return state
    except Exception as e:
        raise Exception(f"Filtering messages error: {e}")


# 用来暂停 langgraph
async def break_point(state: T) -> T:
    logger.info("this is break_point node")
    return state


# 获取用户反馈后的处理
async def human_feedback(state: T) -> T:
    # 这里可以添加逻辑来处理用户反馈
    # 例如，等待用户输入并更新 state["user_feedback"]
    logger.info("this is human_feedback node")
    import rich
    rich.print(state)
    return state


def regist_graph(name: str, input_handler: Type[InputHandler], event_handler: Type[EventHandler]) -> Callable:
    """
    graph 注册工厂类
    :param name: graph 的名称
    :param input_handler: 输入数据结构
    :param event_handler: 输出数据结构
    :return: graph 实例
    """
    def wrapper(func: Callable) -> Callable:
        _GRAPHS_REGISTRY[name] = {
            "func": func,
            "input_handler": input_handler,
            "event_handler": event_handler
        }
        return func
    return wrapper


# 全局字典用于存储不同类型图的名称和对应的类
rag_registry = {}
agent_registry = {}


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

    return cls


class Graph:
    def __init__(self, llm: ChatOpenAI, tools: list[BaseTool], history_len: int):
        self.llm = llm
        self.tools = tools
        self.history_len = history_len

    # async def chatbot(self, state: Type[State]) -> Type[State]:
    #     """
    #     定义了 graph 中 llm 的消息处理逻辑, 子类必须实现.
    #     """
    #     pass

    @abstractmethod
    def get_graph(self) -> CompiledStateGraph:
        """
        定义了 graph 流程, 子类必须实现.
        """
        pass

    @abstractmethod
    async def handle_event(self, *args, **kwargs):
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


def list_graph_titles_by_label(label: str) -> list[str]:
    if label == "rag":
        return [info["title"] for info in rag_registry.values()]
    elif label == "agent":
        return [info["title"] for info in agent_registry.values()]
    else:
        raise ValueError(f"Unknown label '{label}'.")


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
