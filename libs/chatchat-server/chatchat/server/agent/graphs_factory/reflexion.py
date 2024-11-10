import datetime
from typing import List, Optional, Literal, Any

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_openai.chat_models import ChatOpenAI

from chatchat.server.utils import build_logger, get_tool, add_tools_if_not_exists, get_st_graph_memory
from .graphs_registry import State, register_graph, Graph

logger = build_logger()

num_iterations = 0
# 设置反省轮数
MAX_ITERATIONS = 2


class Reflection(BaseModel):
    missing: Optional[str] = Field(
        None,
        description="Critique of what is missing."
    )
    superfluous: Optional[str] = Field(
        None,
        description="Critique of what is superfluous"
    )


class AnswerQuestion(BaseModel):
    """
    Answer the question.
    Provide an answer, reflection, and then follow up with search queries to improve the answer.
    """

    answer: Optional[str] = Field(
        None,
        description="Your answer to the question."
    )
    reflection: Optional[Reflection] = Field(
        None,
        description="Your reflection on the initial answer."
    )
    search_queries: Optional[List[str]] = Field(
        None,
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """
    Revise your original answer to your question. Provide an answer, reflection,
    cite your reflection with references, and finally
    add search queries to improve the answer.
    """

    references: Optional[List[str]] = Field(
        None,
        description="Citations motivating your updated answer."
    )


class ReflexionState(State):
    """
    1. question: 用户的问题.
    2. answer: 答案.
    3. reflection: Your reflection on the initial answer.
    4. search_queries: 1-3 search queries for researching improvements to address the critique of your current answer.
    5. references: Citations motivating your updated answer.
    """
    question: Optional[BaseMessage]
    answer: Optional[str]
    reflection: Optional[Reflection]
    search_queries: Optional[List[str]]
    references: Optional[List[str]]


def extract_messages(messages):
    result = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            result.append(message)
        elif isinstance(message, AIMessage):
            result.append(message)
            break
    return result[::-1]


@register_graph
class ReflexionGraph(Graph):
    name = "reflexion"
    label = "agent"
    title = "自我反思机器人[Beta]"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int):
        super().__init__(llm, tools, history_len)
        # 担心用户没有选择任何 tool 而造成 agent 逻辑无效, 为保证效果, 强行追加一个 search_internet 工具, 如开发者不需要可注释此行代码.
        search_internet = get_tool(name="search_internet")
        self.tools = add_tools_if_not_exists(tools_provides=self.tools, tools_need_append=[search_internet])
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    # tool_node
    async def function_call(self, state: ReflexionState) -> ReflexionState:
        # init
        state["question"] = state["history"][-1]
        if "search_queries" not in state or len(state["search_queries"]) == 0:
            state["search_queries"] = []
            for history_message in state["history"]:
                state["search_queries"].append(history_message)

        tool_node_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an advanced AI capable of identifying and calling appropriate functions to solve various problems.
                    You can call multiple functions simultaneously or the same function multiple times as needed.
                    
                    Current time: {time}
                    
                    Below is a list of questions. For each question, identify the appropriate function(s), call them, and return the results in the order of the questions.
                    If no suitable function is found, return 'No suitable function found'.
                    
                    Questions:
                    {search_queries}
                    """
                ),
            ]
        ).partial(time=lambda: datetime.datetime.now().isoformat())

        llm_with_tools = tool_node_template | self.llm_with_tools

        func_call = llm_with_tools.invoke(state)
        state["messages"] = [func_call]
        state["history"].append(func_call)

        return state

    @staticmethod
    async def process_func_call_history(state: ReflexionState) -> ReflexionState:
        # ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果

        # 找到从末尾开始的第一个非ToolMessage的索引
        index = len(state["messages"]) - 1
        while index >= 0 and isinstance(state["messages"][index], ToolMessage):
            index -= 1

        # index现在指向最后一个非ToolMessage或者-1（如果所有消息都是ToolMessage）
        # 将index后面的所有ToolMessage追加到history中
        # 注意这里使用index + 1来开始切片，因为index指向的是最后一个非ToolMessage
        tool_messages_to_add = state["messages"][index + 1:]
        if tool_messages_to_add:
            state["history"].extend(tool_messages_to_add)
        return state

    @staticmethod
    async def process_reflexion_result(state: ReflexionState, result) -> ReflexionState:
        if hasattr(result, 'reflection') and result.reflection is not None:
            if not isinstance(result.reflection, dict):
                result.reflection = result.reflection.dict()

        if hasattr(result, 'search_queries') and result.search_queries is not None:
            if not isinstance(result.search_queries, list):
                result.search_queries = list(result.search_queries)

        state["messages"].append(AIMessage(content=str(result)))
        state["history"].append(AIMessage(content=str(result)))
        state["answer"] = result.answer
        state["reflection"] = result.reflection
        state["search_queries"] = result.search_queries
        if hasattr(result, 'references'):
            state["references"] = result.references

        return state

    async def initial(self, state: ReflexionState) -> ReflexionState:
        initial_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert researcher.
                    Current time: {time}
                    User's question: {question}

                    Actions and Results so far:
                    {history}
                    
                    Steps:
                    1. Provide correct and concise answers.
                    2. Reflect and critique your initial answer based on the user's question, the actions and results so far, and the reflection. Be thorough to maximize improvement.
                       - For the `missing` field: Identify and describe any important information or perspectives that are absent from the initial answer. For example, if the question is about the benefits of a healthy diet and the initial answer doesn't mention mental health benefits, this should be noted.
                       - For the `superfluous` field: Identify and describe any information or details in the initial answer that are unnecessary or irrelevant. For example, if the initial answer includes a lengthy history of dietary guidelines that doesn't directly address the question, this should be noted.
                    3. List any additional information that needs to be collected. Provide requirements only, no execution needed.
                    4. Provide an updated response using the {function_name} struct. This should include:
                       - Updated Answer: Incorporate insights from the actions and results so far, initial answer, and reflection.
                       - Updated Reflection: Reflect on the new answer, ensuring to fill in the `missing` and `superfluous` fields accurately.
                         - For the `missing` field: Ensure all critical information and perspectives are included.
                         - For the `superfluous` field: Ensure all unnecessary or irrelevant information is removed.
                       - Search Queries: List any additional information that needs to be collected.
                       - References: Include any supporting references used.
                    """,
                ),
            ]
        ).partial(
            time=lambda: datetime.datetime.now().isoformat(),
        )

        initial_answer_chain = initial_prompt_template.partial(function_name=AnswerQuestion.__name__) | self.llm.with_structured_output(AnswerQuestion)

        initial_result = initial_answer_chain.invoke(state)

        if hasattr(initial_result, 'reflection') and initial_result.reflection is not None:
            if not isinstance(initial_result.reflection, dict):
                initial_result.reflection = initial_result.reflection.dict()

        return await self.process_reflexion_result(state, initial_result)

    # Revision
    async def revision(self, state: ReflexionState) -> ReflexionState:
        global num_iterations
        if num_iterations == 0:
            state["references"] = []

        # Initial responder
        revision_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert researcher.
                    Current time: {time}
                    User's question: {question}

                    Actions and Results so far:
                    {history}

                    Initial Answer:
                    {answer}

                    Reflection on the Initial Answer:
                    {reflection}

                    Steps:
                    1. Revise your previous answer using the new information.
                        - Use the previous critique to add important information to your answer.
                        - Include numerical citations in your revised answer to ensure it can be verified.
                        - If you use any external sources or functions to gather information, add a "References" section at the bottom of your answer (which does not count towards the word limit) in the following format:
                            - https://example.com
                            - https://example.com
                        - If no external sources or functions are used, include a "References" section with an empty list.
                        - Use the previous critique to remove superfluous information from your answer.
                        - Ensure the revised answer is clear, concise, and well-structured.
                    2. Reflect and critique your initial answer based on the user's question, the actions and results so far, and the reflection. Be thorough to maximize improvement.
                       - For the `missing` field: Identify and describe any important information or perspectives that are absent from the initial answer. For example, if the question is about the benefits of a healthy diet and the initial answer doesn't mention mental health benefits, this should be noted.
                       - For the `superfluous` field: Identify and describe any information or details in the initial answer that are unnecessary or irrelevant. For example, if the initial answer includes a lengthy history of dietary guidelines that doesn't directly address the question, this should be noted.
                    3. List any additional information that needs to be collected. Provide requirements only, no execution needed.
                    4. Provide an updated response using the {function_name} struct. This should include:
                       - Updated Answer: Incorporate insights from the actions and results so far, initial answer, and reflection.
                       - Updated Reflection: Reflect on the new answer, ensuring to fill in the `missing` and `superfluous` fields accurately.
                         - For the `missing` field: Ensure all critical information and perspectives are included.
                         - For the `superfluous` field: Ensure all unnecessary or irrelevant information is removed.
                       - Search Queries: List any additional information that needs to be collected.
                       - References: Include any supporting references used.

                    Supporting References:
                    {references}
                    """,
                ),
            ]
        ).partial(
            time=lambda: datetime.datetime.now().isoformat(),
        )

        revision_chain = revision_prompt_template.partial(function_name=ReviseAnswer.__name__) | self.llm.with_structured_output(ReviseAnswer)

        revised_result = revision_chain.invoke(state)

        num_iterations += 1

        if hasattr(revised_result, 'reflection') and revised_result.reflection is not None:
            if not isinstance(revised_result.reflection, dict):
                revised_result.reflection = revised_result.reflection.dict()

        return await self.process_reflexion_result(state, revised_result)

    @staticmethod
    async def event_loop(*args: Any, **kwargs: Any) -> Literal["function_call_loop", "__end__"]:
        # in our case, we'll just stop after N plans
        global MAX_ITERATIONS, num_iterations
        if num_iterations >= MAX_ITERATIONS:
            return END
        return "function_call_loop"

    def get_graph(self) -> CompiledGraph:
        """
        description: https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/
        """

        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")
        if not all(isinstance(tool, BaseTool) for tool in self.tools):
            raise TypeError("All items in tools must be instances of BaseTool")

        memory = get_st_graph_memory()

        tool_node = ToolNode(tools=self.tools)

        # Sub Graph
        function_call_sub_graph_builder = StateGraph(ReflexionState)
        function_call_sub_graph_builder.add_node("function_call", self.function_call)
        function_call_sub_graph_builder.add_node("tools", tool_node)
        function_call_sub_graph_builder.add_node("process_func_call_history", self.process_func_call_history)
        function_call_sub_graph_builder.set_entry_point("function_call")
        function_call_sub_graph_builder.add_edge("function_call", "tools")
        function_call_sub_graph_builder.add_edge("tools", "process_func_call_history")
        function_call_sub_graph_builder.add_edge("process_func_call_history", END)
        function_call_sub_graph = function_call_sub_graph_builder.compile(checkpointer=memory)

        # Construct Graph
        builder = StateGraph(ReflexionState)

        builder.add_node("history_manager", self.async_history_manager)
        builder.add_node("draft", self.initial)
        builder.add_node("revise", self.revision)
        builder.add_node("function_call", function_call_sub_graph)
        builder.add_node("function_call_loop", function_call_sub_graph)

        builder.set_entry_point("history_manager")
        builder.add_edge("history_manager", "function_call")
        builder.add_edge("function_call", "draft")
        # draft -> execute_tools
        builder.add_edge("draft", "function_call_loop")
        # execute_tools -> revise
        builder.add_edge("function_call_loop", "revise")
        # revise -> execute_tools OR end
        builder.add_conditional_edges("revise", self.event_loop)

        graph = builder.compile(checkpointer=memory)

        return graph

    @staticmethod
    async def handle_event(node: str, event: ReflexionState) -> Any:
        """
        event example:
        """
        if node == "revise":
            revise_answer = ReviseAnswer(
                question=event["question"],
                answer=event["answer"],
                reflection=event["reflection"],
                search_queries=event["search_queries"],
                references=event["references"]
            )
            return revise_answer
        elif node == "draft":
            answer_question = AnswerQuestion(
                question=event["question"],
                answer=event["answer"],
                reflection=event["reflection"],
                search_queries=event["search_queries"],
            )
            return answer_question
        elif node == "function_call" or node == "function_call_loop":
            function_call = extract_messages(event["messages"])
            return function_call
        return None
