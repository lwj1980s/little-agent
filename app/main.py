from contextlib import asynccontextmanager

import psycopg

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

from app.core.config import settings
from app.agent.graph import build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):

    conn = psycopg.connect(
        settings.DATABASE_URL,
        autocommit=True
    )

    checkpointer = PostgresSaver(conn)

    # 初始化 Checkpoint 数据表
    checkpointer.setup()

    app.state.agent = build_graph(checkpointer)

    yield

    conn.close()


app = FastAPI(
    title="Enterprise Agent",
    lifespan=lifespan
)


class ChatRequest(BaseModel):

    user_id: str = Field(min_length=1, max_length=128)

    session_id: str = Field(min_length=1, max_length=128)

    message: str = Field(min_length=1, max_length=4000)


@app.post("/chat")
def chat(
    request: ChatRequest,
    http_request: Request
):

    # 演示版本使用请求中的 user_id。
    # 生产环境必须从经过验证的 JWT 身份中获取 user_id。
    user_id = request.user_id

    thread_id = f"{user_id}:{request.session_id}"

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 12
    }

    try:

        result = http_request.app.state.agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=request.message
                    )
                ],
                "user_id": user_id,
                "memory_context": "",
                "current_user_message": request.message
            },
            config=config
        )

        return {
            "user_id": user_id,
            "session_id": request.session_id,
            "answer": result["messages"][-1].content
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


@app.get("/health")
def health():
    return {"status": "ok"}