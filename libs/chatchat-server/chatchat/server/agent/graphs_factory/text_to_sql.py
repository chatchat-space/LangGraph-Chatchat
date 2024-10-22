from typing import Union, Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from chatchat.server.utils import build_logger, get_st_graph_memory
from .graphs_registry import regist_graph, InputHandler, EventHandler, State, async_history_manager

logger = build_logger()


class BaseGraphEventHandler(EventHandler):
    def __init__(self):
        pass

    def handle_event(self, node: str, event: State) -> BaseMessage:
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
        return event["messages"][0]


@regist_graph(name="text_to_sql",
              input_handler=InputHandler,
              event_handler=BaseGraphEventHandler)
def text_to_sql(llm: ChatOpenAI, tools: list[BaseTool], history_len: int) -> CompiledStateGraph:
    """
    description: text to sql, only select
    """
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise TypeError("All items in tools must be instances of BaseTool")

    memory = get_st_graph_memory()

    graph_builder = StateGraph(State)

    async def history_manager(state: State) -> State:
        state = await async_history_manager(state, history_len)
        return state

    async def sql_executor(state: State) -> State:
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

        sql_prompt = ChatPromptTemplate.from_template(
            """你是一个智能数据库查询机器人, 擅长于执行 text to sql 任务, 即: 根据数据库中表和字段的作用以及用户的需求, 来生成准确高效的SQL查询语句.
            如下是库表信息和字段代表的内容:
            # 库:作用
            tencent_hub:储存了 csighub 的关系型数据

            ## 表:作用
            auth:储存了 csighub 的用户密码和加密信息
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            user_id:bigint:用户id
            password:varchar(255):用户密码(这里不要返回给用户)
            username:varchar(255):用户名
            salt:varchar(255):加密信息(这里不要返回给用户)

            ## 表:作用
            docker_image:储存了 csighub 的镜像的数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            repository_id:bigint:镜像仓库 id
            namespace:varchar(255):命名空间
            repository_name:varchar(255):镜像仓库名称
            tag:varchar(128):镜像标签
            image_id:varchar(128):镜像 id, 内容为 sha256:xxxx
            digest:varchar(128):摘要信息, 内容为 sha256:xxxx
            size:bigint:大小
            config:mediumtext:镜像配置信息
            previous_digest:varchar(128):上一个摘要信息, 内容为 sha256:xxxx
            last_push_time:timestamp:上一次推送时间
            author:varchar(64):镜像作者
            is_component:tinyint(1):已作废
            push_count:bigint:镜像推送次数
            pull_count:bigint:镜像下载次数
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记
            manifest:mediumtext:镜像层文件信息

            ## 表:作用
            namespace:储存了 csighub 的命名空间数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            name:varchar(255):命名空间名称
            user_id:bigint:用户 id
            organization_id:bigint:csighub 组织 id
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记

            ## 表:作用
            organization:储存了 csighub 的组织数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            name:varchar(255):组织名称
            user_id:bigint:用户 id
            team_id:bigint:团队 id
            title:varchar(64):组织标题
            url:varchar(1024):组织链接
            email:varchar(128):组织邮箱
            summary:text:组织汇总信息
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记

            ## 表:作用
            repository:储存了 cisghub 的镜像仓库数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            name:varchar(255):镜像仓库名称
            user_id:bigint:用户id
            organization_id:bigint:组织id
            description:mediumtext:镜像仓库描述
            is_public:tinyint(1):是否公开, 1 是公开, 0 是私有
            labels:varchar(1024):镜像仓库标签
            namespace:varchar(255):命名空间
            summary:text:信息
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记
            has_component:tinyint(1):是否为 component

            ## 表:作用
            repository_state:储存了 csighub 的镜像仓库状态数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            repository_id:bigint:镜像仓库 id
            docker_pull_count:bigint:镜像拉取次数
            docker_push_count:bigint:镜像推送次数

            ## 表:作用
            repository_team:储存了 csighub 的镜像仓库的所属用户团队数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            repository_id:bigint:镜像仓库 id
            team_id:bigint:团队 id

            ## 表:作用
            session:储存了登陆用户的 session 数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            expire_at:timestamp:过期时间
            key:varchar(36):session 密钥(这里不要返回给用户)
            user_id:bigint:用户 id

            ## 表:作用
            team:储存了 csighub 的用户团队数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            organization_id:bigint:组织 id
            title:varchar(64):标题
            summary:text:信息汇总

            ## 表:作用
            team_permission:储存了 csighub 的用户团队权限数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            team_id:bigint:团队id
            resource_desc:varchar(128):团队资源描述
            role_name:char(20):团队角色
            resource_id:varchar(32):团队资源id
            description:varchar(128):团队权限描述

            ## 表:作用
            team_user:储存了 csighub 的团队中用户的数据
            ### 字段:类型:作用
            id:int:id
            created_at:timestamp:创建时间
            updated_at:timestamp:更新时间
            team_id:bigint:团队 id
            user_id:bigint:用户 id

            ## 表:作用
            user:储存了 csighub 的用户数据
            ### 字段:类型:作用
            id:int:用户id
            created_at:timestamp:用户创建时间
            updated_at:timestamp:用户更新时间
            name:varchar(255):用户名 

            ## 表:作用
            user_permission:
            ### 字段:类型:作用
            id:int:用户id
            created_at:timestamp:用户创建时间
            updated_at:timestamp:用户更新时间
            user_id:bigint:用户 id
            resource_desc:varchar(128):用户资源描述
            role_name:char(20):用户角色
            resource_id:varchar(32):用户权限 id
            description:varchar(128):用户权限描述

            用户问题:
            {history}

            要求:
            1.只生成查询(SELECT)语句, 其他涉及改数据的需求不允许操作;
            2.请你严格审视你提供的 SQL 是否正确, 因为数据类问题的结果在实际生产环境下非常重要, 不容有失;
            3.答案尽可能以表格的形式返回.
            4.如果用户的问题实在与你所掌握的库表信息无关或非数据查询类问题, 你也可以化身为聊天机器人直接返回用户的问题, 但是前提是你一定要优先考虑用户问题是否与数据库表查询有关.

            举例:
            1.需求: 查询组织`tcs_public`的用户都有谁?
            SQL: `SELECT o.id AS organization_id, o.name AS organization_name, tu.created_at AS team_user_created_at, tu.updated_at AS team_user_updated_at, tu.team_id AS team_id, u.id AS user_id, u.name AS user_name FROM organization o JOIN team_user tu ON o.team_id = tu.team_id JOIN user u ON tu.user_id = u.id WHERE o.name = 'tcs_public';`
            返回:
            +-----------------+-------------------+----------------------+----------------------+---------+---------+-------------+
            | organization_id | organization_name | team_user_created_at | team_user_updated_at | team_id | user_id | user_name   |
            +-----------------+-------------------+----------------------+----------------------+---------+---------+-------------+
            |            9242 | tcs_public        | 2022-08-19 15:08:34  | 2022-08-19 15:08:34  |   10511 |    6620 | dickonliu   |
            |            9242 | tcs_public        | 2022-08-23 11:30:59  | 2022-08-23 11:30:59  |   10511 |   16832 | yuehuazhang |
            +-----------------+-------------------+----------------------+----------------------+---------+---------+-------------+
            """
        )

        llm_with_tools = sql_prompt | llm.bind_tools(tools)

        messages = llm_with_tools.invoke(state)
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)
        return state

    async def result_synthesizer(state: State) -> State:
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

        result_synthesizer_prompt = ChatPromptTemplate.from_template(
            """你是一个智能数据整合机器人, 可以将数据查询结果以 Markdown 格式的表格返回.
            对话历史记录:
            {history}
            """
        )

        llm_result_synthesizer = result_synthesizer_prompt | llm

        messages = llm_result_synthesizer.invoke(state)
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)
        return state

    # def tools_condition(
    #         state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    # ) -> Literal["tools", "__end__"]:
    #     if isinstance(state, list):
    #         ai_message = state[-1]
    #     elif isinstance(state, dict) and (messages := state.get("messages", [])):
    #         ai_message = messages[-1]
    #     elif messages := getattr(state, "messages", []):
    #         ai_message = messages[-1]
    #     else:
    #         raise ValueError(f"No messages found in input state to tool_edge: {state}")
    #     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
    #         return "tools"
    #     return "__end__"

    tool_node = ToolNode(tools=tools)

    graph_builder.add_node("history_manager", history_manager)
    graph_builder.add_node("sql_executor", sql_executor)
    graph_builder.add_node("result_synthesizer", result_synthesizer)
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("history_manager")
    graph_builder.add_edge("history_manager", "sql_executor")
    graph_builder.add_conditional_edges(
        "sql_executor",
        tools_condition,
    )
    # graph_builder.add_edge("sql_executor", "tools")
    graph_builder.add_edge("tools", "result_synthesizer")
    graph_builder.add_edge("result_synthesizer", END)

    graph = graph_builder.compile(checkpointer=memory)

    return graph
