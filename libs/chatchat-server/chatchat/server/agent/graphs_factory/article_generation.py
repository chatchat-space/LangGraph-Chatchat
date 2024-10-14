import rich
from typing import Optional, Annotated, List, TypedDict, Any
from pydantic import BaseModel, Field

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import (
    build_logger,
    get_st_graph_memory,
    get_tool,
    create_agent_models
)
from .graphs_registry import (
    regist_graph,
    InputHandler,
    EventHandler,
)

logger = build_logger()


class Article(BaseModel):
    """Respond to the user with this"""
    article: str = Field(description="The content of article")


class ArticleListResponse(BaseModel):
    """Respond to the user with this"""
    article_list: List[str] = Field(description="The list of article")


class ArticleGenerationState(TypedDict):
    """
    定义一个基础 State 供 各类 graph 继承, 其中:
    1. messages 为所有 graph 的核心信息队列, 所有聊天工作流均应该将关键信息补充到此队列中;
    2. history 为所有工作流单次启动时获取 history_len 的 messages 所用(节约成本, 及防止单轮对话 tokens 占用长度达到 llm 支持上限),
    history 中的信息理应是可以被丢弃的.
    """
    messages: Annotated[List[BaseMessage], add_messages]  # 消息队列
    article_links: Optional[str]  # 待爬取文章链接
    image_links: Optional[str]  # 图片链接
    article_links_list: Optional[list[str]]  # 待爬取文章链接列表
    image_links_list: Optional[list[str]]  # 图片链接列表
    article_list: Optional[list[str]]  # 已爬取的文章内容列表
    llm: Optional[str]  # 选择的 llm
    temperature: Optional[float]  # 选择的 temperature
    article: Optional[str]  # 生成的文章
    user_prompt: Optional[str]  # 用户指令
    is_article_generation_complete: Optional[bool]  # 消息队列


# 用来暂停 langgraph, 使得用户传入 待爬文章链接 和 图片链接
async def article_generation_init_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_INIT_break_point node and now i will break this graph.")
    return state


# 用来暂停 langgraph, 使得用户传入 指令
async def article_generation_start_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_START_break_point node and now i will break this graph.")
    return state


# 用来暂停 langgraph, 使得用户传入 重写指令
async def article_generation_repeat_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_REPEAT_break_point node and now i will break this graph.")
    return state


# 用户传入待爬文章链接后的处理
async def url_handler(state: ArticleGenerationState) -> ArticleGenerationState:
    # 处理用户输入的文章和图片链接
    article_links_list = [link.strip() for link in state["article_links"].split('\n') if link.strip()]
    image_links_list = [link.strip() for link in state["image_links"].split('\n') if link.strip()]
    state["article_links_list"] = article_links_list
    state["image_links_list"] = image_links_list

    call_spider_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a web crawler AI designed to efficiently extract article content from specified web links using the crawler tool (url_reader).
    
    Instructions:
    1. Analyze the provided list of article links and determine if the crawler tool needs to be called once for a single link or multiple times for several links.
    2. Crawl the articles from the following links:
    {article_links_list}
    
    Requirements:
    - Identify and remove any advertising-related content, promotional material, or unrelated sections from the articles.
    - Return only the main content of each article without summarizing, rewriting, or altering the text.
    - Ensure that the final output is a list of strings, where each string represents the complete content of an individual article.
    
    Additional Considerations:
    - If a link leads to a page that cannot be crawled or does not contain valid article content, return a placeholder string indicating the issue (e.g., "Unable to retrieve content from [link]").
    - Maintain the order of the links in the output list to correspond with the input list.
    
    Final Output:
    - A list of strings containing the cleaned article content, with each element corresponding to the respective link in the input list.
    """
            ),
        ]
    )
    prompt = call_spider_template.format(article_links_list=state["article_links_list"])
    state["messages"] = [HumanMessage(content=prompt)]

    return state


async def article_writer(state: ArticleGenerationState) -> ArticleGenerationState:
    writer_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are an article-writing robot skilled in creating various types of articles (news, reports, entertainment, sports, technology, etc.). Readers enjoy your writing. I will provide you with a list of article contents. Please process them according to the instructions below.
    
    Article content list:
    {article_list}
    
    Instructions:
    {user_prompt}
    
    Final output:
    Process the articles as instructed and return the results.
    """
            ),
        ]
    )
    writer_llm = create_agent_models(configs=None,
                                     model=state["llm"],
                                     max_tokens=None,
                                     temperature=state["temperature"],
                                     stream=True)
    print("--- writer ---")
    rich.print(writer_llm)
    llm_with_structured_output = writer_template | writer_llm.with_structured_output(Article)
    writer_llm_result = llm_with_structured_output.invoke(state)
    # 先把用户的指令追加到 messages 消息队列中
    state["messages"].append(HumanMessage(content=state["user_prompt"]))
    state["messages"].append(AIMessage(content=str(writer_llm_result["article"])))
    state["article"] = writer_llm_result["article"]
    return state


async def article_rewriter(state: ArticleGenerationState) -> ArticleGenerationState:
    rewriter_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are an article-writing robot skilled in creating various types of articles (news, reports, entertainment, sports, technology, etc.). Readers enjoy your writing. I will provide you with a list of article contents. Please process them according to the instructions below.

    Article content:
    {article}

    Instructions:
    {user_prompt}

    Final output:
    Process the articles as instructed and return the results.
    """
            ),
        ]
    )
    rewriter_llm = create_agent_models(configs=None,
                                       model=state["llm"],
                                       max_tokens=None,
                                       temperature=state["temperature"],
                                       stream=True)
    print("--- rewriter ---")
    rich.print(rewriter_llm)
    llm_with_structured_output = rewriter_template | rewriter_llm.with_structured_output(Article)
    rewriter_llm_result = llm_with_structured_output.invoke(state)
    # 先把用户的指令追加到 messages 消息队列中
    state["messages"].append(HumanMessage(content=state["user_prompt"]))
    state["messages"].append(AIMessage(content=str(rewriter_llm_result["article"])))
    state["article"] = rewriter_llm_result["article"]
    return state


async def should_continue(state: ArticleGenerationState):
    # If there is no function call, then we respond to the user
    if state["is_article_generation_complete"]:
        return "article_generation_complete"
    # Otherwise if there is, we continue
    else:
        return "article_generation_not_complete"


# 用来爬取用户输入的网页文章内容
def get_spider_graph(llm: ChatOpenAI, memory: Any) -> CompiledStateGraph:
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")

    url_reader = get_tool(name="url_reader")
    tools = [url_reader]
    llm_with_tools = llm.bind_tools(tools)

    async def function_call(state: ArticleGenerationState) -> ArticleGenerationState:
        llm_result = llm_with_tools.invoke(state["messages"])
        state["messages"] = [llm_result]
        return state

    respond_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    Requirements:
    - Return only the content of each article crawled by the url_reader tool above, without summarizing, rewriting, or modifying the text in any way.
    - The output should not contain any additional comments, explanations, or formatting changes.
    - Ensure that the final output is a list of strings, where each string represents the complete content of a single article from the input list.
    - The article crawled by the url_reader list:
    {messages}
    
    Final output:
    - A list of strings containing the complete content of each article. Each element in the list should correspond to an article in the input, leaving the original text intact.
    """
            ),
        ]
    )
    llm_with_structured_output = respond_template | llm.with_structured_output(ArticleListResponse)

    # Define the function that responds to the user
    async def make_article_list(state: ArticleGenerationState) -> ArticleGenerationState:
        # We call the model with structured output in order to return the same format to the user every time
        # state['messages'][-2] is the last ToolMessage in the convo, which we convert to a HumanMessage for the model to use
        # We could also pass the entire chat history, but this saves tokens since all we care to structure is the output of the tool
        llm_result = llm_with_structured_output.invoke(state)
        for r in llm_result["article_list"]:
            state["messages"].append(AIMessage(content=r))
        state["article_list"] = llm_result["article_list"]
        # We return the final answer
        return state

    # Define the function that determines whether to continue or not
    async def should_continue(state: ArticleGenerationState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we respond to the user
        if not last_message.tool_calls:
            return "make_article_list"
        # Otherwise if there is, we continue
        else:
            return "continue"

    graph_builder = StateGraph(ArticleGenerationState)

    tool_node = ToolNode(tools=tools)

    graph_builder.add_node("function_call", function_call)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("make_article_list", make_article_list)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    graph_builder.set_entry_point("function_call")

    # We now add a conditional edge
    graph_builder.add_conditional_edges(
        "function_call",
        should_continue,
        {
            "continue": "tools",
            "make_article_list": "make_article_list",
        },
    )

    graph_builder.add_edge("tools", "function_call")
    graph_builder.add_edge("make_article_list", END)

    graph = graph_builder.compile(checkpointer=memory)

    return graph


class ArticleGenerationEventHandler(EventHandler):
    def __init__(self):
        pass

    def handle_event(self, node: str, events: ArticleGenerationState) -> BaseMessage:
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
        return events["messages"][0]


@regist_graph(name="article_generation",
              input_handler=InputHandler,
              event_handler=ArticleGenerationEventHandler)
def article_generation(llm: ChatOpenAI, tools: list[BaseTool], history_len: int) -> CompiledGraph:
    """
    description: https://langchain-ai.github.io/langgraph/tutorials/introduction/
    """
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise TypeError("All items in tools must be instances of BaseTool")

    memory = get_st_graph_memory()

    graph_builder = StateGraph(ArticleGenerationState)

    spider_graph = get_spider_graph(llm=llm, memory=memory)

    graph_builder.add_node("article_generation_init_break_point", article_generation_init_break_point)
    graph_builder.add_node("url_handler", url_handler)
    graph_builder.add_node("spider", spider_graph)
    graph_builder.add_node("article_generation_start_break_point", article_generation_start_break_point)
    graph_builder.add_node("writer", article_writer)
    graph_builder.add_node("article_generation_repeat_break_point", article_generation_repeat_break_point)
    graph_builder.add_node("rewriter", article_rewriter)

    graph_builder.set_entry_point("article_generation_init_break_point")
    graph_builder.add_edge("article_generation_init_break_point", "url_handler")
    graph_builder.add_edge("url_handler", "spider")
    graph_builder.add_edge("spider", "article_generation_start_break_point")
    graph_builder.add_edge("article_generation_start_break_point", "writer")
    graph_builder.add_edge("writer", "article_generation_repeat_break_point")
    graph_builder.add_edge("article_generation_repeat_break_point", "rewriter")
    # We now add a conditional edge
    graph_builder.add_conditional_edges(
        "rewriter",
        should_continue,
        {
            "article_generation_complete": END,
            "article_generation_not_complete": "article_generation_repeat_break_point",
        },
    )

    graph = graph_builder.compile(checkpointer=memory, interrupt_after=[
        "article_generation_init_break_point",
        "article_generation_start_break_point",
        "article_generation_repeat_break_point",
    ])

    return graph
