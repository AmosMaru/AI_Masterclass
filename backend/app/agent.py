# LlamaIndex imports
from app.schema import InputEvent, StreamEvent, ToolCallEvent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from typing import Any, List

# Function calling agent implementation
class FunctionCallingAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or LlamaOpenAI()
        assert self.llm.metadata.is_function_calling_model

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        # clear sources
        await ctx.set("sources", [])

        # check if memory is setup
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # get chat history
        chat_history = memory.get()

        # update context
        await ctx.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        # stream the response
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # save the final response, which should have all content
        memory = await ctx.get("memory")
        memory.put(response.message)
        await ctx.set("memory", memory)

        # get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.get("sources", default=[])
            return StopEvent(
                result={"response": response, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # update memory
        memory = await ctx.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.set("sources", sources)
        await ctx.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)
