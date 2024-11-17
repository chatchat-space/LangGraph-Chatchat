from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_tool, add_tools_if_not_exists
from .graphs_registry import State, register_graph, Graph

logger = build_logger()


@register_graph
class TextToSQLGraph(Graph):
    name = "text_to_sql"
    label = "agent"
    title = "数据库查询机器人[Beta]"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver):
        super().__init__(llm, tools, history_len, checkpoint)
        query_sql_data = get_tool(name="query_sql_data")
        self.tools = add_tools_if_not_exists(tools_provides=self.tools, tools_need_append=[query_sql_data])
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def sql_executor(self, state: State) -> State:
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

        # 下面涉及的库表信息均为测试构造, 仅做参考, 请开发者自行修改.
        sql_prompt = ChatPromptTemplate.from_template(
            """你是一个智能数据库查询助手, 负责将用户需求转换为 SQL 查询语句. 请根据数据库中表和字段的作用以及用户的需求, 生成准确高效的 SQL 查询语句.

            以下是库表信息和核心字段的说明:
            # 库:作用
            docker_hub:储存了 docker_hub 平台的关系型数据

            ## 表:作用
            auth:储存了用户密码和加密数据
            ### 字段:类型:作用
            id:int:auth id
            user_id:bigint:用户 id
            password:varchar(255):用户密码(不返回给用户)
            username:varchar(255):用户名
            salt:varchar(255):加密信息(不返回给用户)

            ## 表:作用
            docker_image:储存了镜像的元数据
            ### 字段:类型:作用
            id:int:id
            repository_id:bigint:镜像仓库 id
            namespace:varchar(255):命名空间
            repository_name:varchar(255):镜像仓库名称
            tag:varchar(128):镜像标签
            image_id:varchar(128):镜像 id, 内容为 sha256:xxxx
            digest:varchar(128):摘要信息, 内容为 sha256:xxxx
            size:bigint:镜像大小
            config:mediumtext:镜像配置信息
            previous_digest:varchar(128):上一个镜像版本的摘要信息, 内容为 sha256:xxxx
            last_push_time:timestamp:最近一次镜像推送时间
            author:varchar(64):镜像作者
            is_component:tinyint(1):已作废
            push_count:bigint:镜像推送次数
            pull_count:bigint:镜像下载次数
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记
            manifest:mediumtext:镜像层文件信息

            ## 表:作用
            namespace:储存了命名空间的元数据
            ### 字段:类型:作用
            id:int:命名空间 id
            name:varchar(255):命名空间名称
            user_id:bigint:用户 id
            organization_id:bigint:组织 id
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记

            ## 表:作用
            organization:储存了组织的元数据(organization != team)
            ### 字段:类型:作用
            id:int:组织 id(注意: organization.id != team.id)
            name:varchar(255):组织名称
            user_id:bigint:用户 id
            team_id:bigint:团队 id(注意: organization.team_id == team.id)
            title:varchar(64):组织标题
            url:varchar(1024):组织链接
            email:varchar(128):组织邮箱
            summary:text:组织汇总信息
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记

            ## 表:作用
            repository:储存了镜像仓库的元数据
            ### 字段:类型:作用
            id:int:镜像仓库 id
            name:varchar(255):镜像仓库名称
            user_id:bigint:用户 id
            organization_id:bigint:组织 id
            description:mediumtext:镜像仓库描述
            is_public:tinyint(1):是否公开, 1 是公开, 0 是私有
            labels:varchar(1024):镜像仓库标签
            namespace:varchar(255):命名空间
            summary:text:信息
            deleted_at:timestamp:删除时间
            deleted_flag:bigint:删除标记

            ## 表:作用
            repository_state:储存了镜像仓库的状态数据
            ### 字段:类型:作用
            id:int:repository_state id
            repository_id:bigint:镜像仓库 id
            docker_pull_count:bigint:镜像拉取次数
            docker_push_count:bigint:镜像推送次数

            ## 表:作用
            repository_team:储存了镜像仓库所属用户团队的数据
            ### 字段:类型:作用
            id:int:repository_team id
            repository_id:bigint:镜像仓库 id
            team_id:bigint:团队 id

            ## 表:作用
            session:储存了登陆用户的 session 数据
            ### 字段:类型:作用
            id:int:session id
            expire_at:timestamp:过期时间
            key:varchar(36):session 密钥(不返回给用户)
            user_id:bigint:用户 id

            ## 表:作用
            team:储存了团队的元数据
            ### 字段:类型:作用
            id:int:团队 id
            organization_id:bigint:组织 id(注意: team.organization_id == organization.id)

            ## 表:作用
            team_permission:储存了团队的权限数据
            ### 字段:类型:作用
            id:int:team_permission id
            team_id:bigint:团队 id
            resource_desc:varchar(128):团队资源描述
            role_name:char(20):团队角色
            resource_id:varchar(32):团队资源 id
            description:varchar(128):团队权限描述

            ## 表:作用
            team_user:储存了团队用户的数据
            ### 字段:类型:作用
            id:int:team_user id
            team_id:bigint:团队 id
            user_id:bigint:用户 id

            ## 表:作用
            user:储存了用户的元数据
            ### 字段:类型:作用
            id:int:用户 id
            name:varchar(255):用户名 

            ## 表:作用
            user_permission:储存了用户权限的数据
            ### 字段:类型:作用
            id:int:user_permission id
            user_id:bigint:用户 id
            resource_desc:varchar(128):用户资源描述
            role_name:char(20):用户角色
            resource_id:varchar(32):用户权限 id
            description:varchar(128):用户权限描述

            用户问题:
            {history}

            注意与要求:
            1. 只允许生成查询（SELECT）语句，其他涉及改数据的 SQL 不允许被执行。
            2. 请严格审视提供的 SQL 是否正确，数据类问题的结果在实际生产环境下非常重要，不容有失。
            3. 在 docker_hub 库中，organization 和 team 是两个维度的概念，请不要将 organization 表的 id 等同于 team 表的 id 或 organization 表的 team_id。
            4. 答案尽可能以表格的形式返回，表格格式应清晰易读。
            5. 如果用户的问题与数据库表查询无关，可以友好地引导用户提出与数据库相关的问题。
            6. 对于无效查询或无法识别的请求，返回一条友好的提示信息。

            示例:
            1.需求: 查询组织`docker_test_org`的用户都有谁?
            SQL: `SELECT o.id AS organization_id, o.name AS organization_name, tu.created_at AS team_user_created_at, tu.updated_at AS team_user_updated_at, tu.team_id AS team_id, u.id AS user_id, u.name AS user_name FROM organization o JOIN team_user tu ON o.team_id = tu.team_id JOIN user u ON tu.user_id = u.id WHERE o.name = 'docker_test_org';`
            返回:
            +-----------------+-------------------+----------------------+----------------------+---------+---------+-------------+
            | organization_id | organization_name | team_user_created_at | team_user_updated_at | team_id | user_id | user_name   |
            +-----------------+-------------------+----------------------+----------------------+---------+---------+-------------+
            |      11111      | docker_test_org   | 1969-08-19 15:08:34  | 1969-08-19 15:08:34  |   2222  |   1212  |  test_user  |
            |      11111      | docker_test_org   | 1969-08-23 11:30:59  | 1969-08-23 11:30:59  |   2222  |   2323  |  yuehua-s   |
            +-----------------+-------------------+----------------------+----------------------+---------+---------+-------------+

            2.需求: 查询用户`yuehua-s`的信息
            SQL: `SELECT * FROM user WHERE name='yuehua-s';`
            返回:
            +-------+---------------------+---------------------+-------------+
            | id    | created_at          | updated_at          | name        |
            +-------+---------------------+---------------------+-------------+
            | 2323  | 1969-03-08 17:32:03 | 1969-03-08 17:32:03 |   yuehua-s  |
            +-------+---------------------+---------------------+-------------+
            """
        )

        llm_with_tools = sql_prompt | self.llm_with_tools

        messages = llm_with_tools.invoke(state)
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)

        return state

    async def result_synthesizer(self, state: State) -> State:
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

        llm_result_synthesizer = result_synthesizer_prompt | self.llm

        messages = llm_result_synthesizer.invoke(state)
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)

        return state

    def get_graph(self) -> CompiledStateGraph:
        """
        description: text to sql, only select
        """
        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")
        if not all(isinstance(tool, BaseTool) for tool in self.tools):
            raise TypeError("All items in tools must be instances of BaseTool")

        graph_builder = StateGraph(State)

        tool_node = ToolNode(tools=self.tools)

        graph_builder.add_node("history_manager", self.async_history_manager)
        graph_builder.add_node("sql_executor", self.sql_executor)
        graph_builder.add_node("result_synthesizer", self.result_synthesizer)
        graph_builder.add_node("tools", tool_node)

        graph_builder.set_entry_point("history_manager")
        graph_builder.add_edge("history_manager", "sql_executor")
        graph_builder.add_conditional_edges(
            "sql_executor",
            tools_condition,
        )
        graph_builder.add_edge("tools", "result_synthesizer")
        graph_builder.add_edge("result_synthesizer", END)

        graph = graph_builder.compile(checkpointer=self.checkpoint)

        return graph

    @staticmethod
    def handle_event(node: str, event: State) -> BaseMessage:
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
        return event["messages"][-1]