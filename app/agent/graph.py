from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.agent.nodes import (
    load_memory_node,
    agent_node,
    save_memory_node
)

from app.agent.tools import TOOLS


def route_agent(state: AgentState):

    last_message = state["messages"][-1]

    # 如果 LLM 请求调用工具
    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"

    # 否则保存记忆并结束
    return "save_memory"


def build_graph(checkpointer):

    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node(
        "load_memory",
        load_memory_node
    )

    builder.add_node(
        "agent",
        agent_node
    )

    builder.add_node(
        "tools",
        ToolNode(TOOLS)
    )

    builder.add_node(
        "save_memory",
        save_memory_node
    )

    # 设置入口
    builder.add_edge(
        START,
        "load_memory"
    )

    builder.add_edge(
        "load_memory",
        "agent"
    )

    # Agent 条件路由
    builder.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "save_memory": "save_memory"
        }
    )

    # 工具执行完成后重新交给 Agent
    builder.add_edge(
        "tools",
        "agent"
    )

    # 保存记忆后结束
    builder.add_edge(
        "save_memory",
        END
    )

    return builder.compile(
        checkpointer=checkpointer
    )