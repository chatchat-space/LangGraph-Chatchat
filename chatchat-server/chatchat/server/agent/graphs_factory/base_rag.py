from typing import List, Literal, Dict

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, filter_messages, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import (
    build_logger,
    get_tool,
    add_tools_if_not_exists,
)
from .graphs_registry import State, Graph, register_graph

logger = build_logger()


class BaseRagState(State):
    """
    定义一个基础 State 供 各类 rag graph 继承, 其中:
    1. messages 为所有 graph 的核心信息队列, 所有聊天工作流均应该将关键信息补充到此队列中;
    2. history 为所有工作流单次启动时获取 history_len 的 messages 所用(节约成本, 及防止单轮对话 tokens 占用长度达到 llm 支持上限),
    history 中的信息理应是可以被丢弃的.
    """
    knowledge_base: str
    top_k: int
    score_threshold: float
    question: str
    docs: List[Dict]
    retrieve_retry: int


@register_graph
class BaseRagGraph(Graph):
    name = "base_rag"
    label = "rag"
    title = "基础RAG"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver,
                 knowledge_base: str,
                 top_k: int,
                 score_threshold: float):
        super().__init__(llm, tools, history_len, checkpoint)
        search_local_knowledgebase = get_tool(name="search_local_knowledgebase")
        self.tools = add_tools_if_not_exists(tools_provides=self.tools, tools_need_append=[search_local_knowledgebase])
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def async_history_manager(self, state: BaseRagState) -> BaseRagState:
        """
        目的: 1. 节约成本; 2. 初始化 state
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
            state["question"] = state["history"][-1].content
            state["knowledge_base"] = self.knowledge_base
            state["top_k"] = self.top_k
            state["score_threshold"] = self.score_threshold
            return state
        except Exception as e:
            raise Exception(f"Filtering messages error: {e}")

    async def chatbot(self, state: BaseRagState) -> BaseRagState:
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        # ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果, 否则报错如下:
        # Error code: 400 -
        # {
        #     "error": {
        #         "message": "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
        #         "type": "invalid_request_error",
        #         "param": "messages.[1].role",
        #         "code": null
        #     }
        # }
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])

        prompt = PromptTemplate(
            template="""
            You are an intelligent robot that determines whether it is necessary to call the local knowledge base query tool to answer questions.

            The knowledge base call parameters are as follows.

            knowledge_base:
            {knowledge_base}

            top_k:
            {top_k}

            score_threshold:
            {score_threshold}

            The chat history and user questions are as follows:
            {history}
            """,
            input_variables=["history", "knowledge_base", "top_k", "score_threshold"],
        )

        llm_with_tools = prompt | self.llm_with_tools

        message = await llm_with_tools.ainvoke(state)
        state["messages"] = [message]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(message)
        return state

    async def grade_documents(self, state: BaseRagState) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        # Data model
        class Grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # Prompt
        prompt = PromptTemplate(
            template="""
            You are a grader assessing relevance of a retrieved document to a user question. 
            Here is the retrieved document:
            {docs}

            Here is the history and user question: 
            {history}

            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            """,
            input_variables=["docs", "history"],
        )

        # Chain
        referee = prompt | self.llm.with_structured_output(Grade)
        scored_result = await referee.ainvoke(state)
        # score = scored_result.binary_score
        # 检查 scored_result 是否为 None
        if scored_result is None:
            logger.warning(f"The scored_result is None. Defaulting to 'yes'. Question: {state['question']}")
            score = "yes"
        else:
            score = scored_result.binary_score

        if score == "yes":
            return "generate"
        else:
            return "rewrite"

    async def generate(self, state: BaseRagState) -> BaseRagState:
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
             dict: The updated state with re-phrased question
        """
        # Prompt
        # prompt = hub.pull("rlm/rag-prompt")
        # prompt_template = get_prompt_template("rag", "default")
        prompt = PromptTemplate(
            template="""
            【指令】
            根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。
            
            【已知信息】
            {docs}
            
            【历史消息及用户问题】
            {history}
            """,
            input_variables=["context", "question"],
        )

        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()

        # Run
        response = await rag_chain.ainvoke(state)
        state["messages"].append(AIMessage(content=response))
        # state["history"].append(AIMessage(content=response))  # 其实无意义, 因为已经走到 graph 的最后一步

        return state

    async def rewrite(self, state: BaseRagState) -> BaseRagState:
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        prompt = PromptTemplate(
            template="""
            Look at the input and try to reason about the underlying semantic intent / meaning.
            Here is the history:
            {history}
             
            Here is the initial question:
            {question}
             
            Formulate an improved question: 
            """,
            input_variables=["question", "history"],
        )

        llm = prompt | self.llm
        # Grader
        response = await llm.ainvoke(state)
        message = HumanMessage(content=response.content)

        state["messages"] = [message]
        state["history"].append(message)
        state["question"] = response.content

        return state

    def get_graph(self) -> CompiledStateGraph:
        """
        description: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
        """
        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")
        if not all(isinstance(tool, BaseTool) for tool in self.tools):
            raise TypeError("All items in tools must be instances of BaseTool")

        # Define a new graph
        graph_builder = StateGraph(BaseRagState)

        retrieve = ToolNode(tools=self.tools)

        # Define the nodes we will cycle between
        graph_builder.add_node("history_manager", self.async_history_manager)
        graph_builder.add_node("chatbot", self.chatbot)  # agent
        graph_builder.add_node("retrieve", retrieve)  # retrieval
        graph_builder.add_node("init_docs", self.init_docs)
        graph_builder.add_node("rewrite", self.rewrite)  # Re-writing the question
        graph_builder.add_node("generate", self.generate)  # Generating a response after we know the documents are relevant

        # Call chatbot node to decide to retrieve or not
        graph_builder.add_edge(START, "history_manager")
        graph_builder.add_edge("history_manager", "chatbot")
        # Decide whether to retrieve
        graph_builder.add_conditional_edges(
            "chatbot",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )
        graph_builder.add_edge("retrieve", "init_docs")
        # Edges taken after the `action` node is called.
        graph_builder.add_conditional_edges(
            "init_docs",
            # Assess agent decision
            self.grade_documents,
        )
        graph_builder.add_edge("rewrite", "chatbot")
        graph_builder.add_edge("generate", END)

        # Compile
        graph = graph_builder.compile(checkpointer=self.checkpoint)
        return graph

    @staticmethod
    def handle_event(node: str, event: BaseRagState) -> BaseMessage:
        """
        event example:
        {
            'messages': [HumanMessage(
                            content='The youtube video of Xiao Yixian in Fights Break Sphere?',
                            id='b9c5468a-7340-425b-ae6f-2f584a961014')],
            'history': [HumanMessage(
                            content='The youtube video of Xiao Yixian in Fights Break Sphere?',
                            id='b9c5468a-7340-425b-ae6f-2f584a961014')]
        }
        """
        # todo: 讲道理 events = graph.astream(input=graph_input, config=graph_config, stream_mode="updates")
        #  只需要 event["messages"][0], 每次都是更新最新的 "messages", 但是这里不行, 需要再研究一下
        return event["messages"][-1]
