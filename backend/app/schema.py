from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Event,
)
from llama_index.core.tools import ToolSelection, ToolOutput
from pydantic import BaseModel
from typing import List, Optional

# LlamaIndex Event classes
class InputEvent(Event):
    input: list[ChatMessage]

class StreamEvent(Event):
    delta: str

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]

class FunctionOutputEvent(Event):
    output: ToolOutput

# Pydantic models for API
class Message(BaseModel):
    message: Optional[str] = None

class ResponseMessage(BaseModel):
    text: str
    audio: str
    lipsync: dict
    facialExpression: str
    animation: str

class ChatResponse(BaseModel):
    messages: List[ResponseMessage]
