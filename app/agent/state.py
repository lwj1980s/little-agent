from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    # 对话历史
    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # 用户 ID
    user_id: str

    # Mem0 检索出来的长期记忆
    memory_context: str

    # 当前轮用户输入，用于保存本轮记忆
    current_user_message: str