from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os
import json
import base64
import logging
import time
from logging.handlers import RotatingFileHandler
import sys
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

# LlamaIndex imports
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
    Event,
    step,
)


# ------------------------------------------------------------------------------------------
# Configure logging
# ------------------------------------------------------------------------------------------

# Configure logging
def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("ai_masterclass")
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '{"time":"%(asctime)s", "level":"%(levelname)s", "module":"%(module)s", "function":"%(funcName)s", "line":%(lineno)d, "message":"%(message)s"}'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler (rotates at 10MB, keeps 5 backup files)
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()
load_dotenv()
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "-"))
# Voice options for Whisper TTS
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DEFAULT_VOICE = "alloy"


# ------------------------------------------------------------------------------------------
# Define Pydantic models
# ------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------
# Define tools for function calling
# ------------------------------------------------------------------------------------------

# Sample tools for function calling
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is currently sunny with a temperature of 72°F."

def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone."""
    current_time = time.strftime("%H:%M:%S", time.gmtime())
    return f"The current time in {timezone} is {current_time}."

def remember_fact(fact: str) -> str:
    """Remember a fact about the user."""
    return f"I'll remember that {fact}"

# ----------------------------------------------------------------------------------------------
# Define the LlamaIndex agent
# ----------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------
# Initialize tools
# ------------------------------------------------------------------------------------------
tools = [
    FunctionTool.from_defaults(get_weather),
    FunctionTool.from_defaults(get_time),
    FunctionTool.from_defaults(remember_fact),
]

# Global agent and context objects
agent = None
agent_context = None
# -------------------------------------------------------------------------------------------
# Define utility functions
# -------------------------------------------------------------------------------------------

# 1 Execute shell command
async def execute_command(command: str) -> str:
    """Execute a shell command asynchronously and return the output for generating lip sync data."""
    logger.debug(f"Executing command: {command}")
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"Command failed with error: {error_msg}")
            raise Exception(f"Command failed: {error_msg}")
        logger.debug("Command executed successfully")
        return stdout.decode()
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        raise

# 2. Convert MP3 to WAV and generate lip sync data
async def lip_sync_message(message_num: int) -> None:
    """
    Convert MP3 to WAV and generate lip sync data using Rhubarb,
    Rhubarb is a tool for generating lip sync data from audio files.
    """
    start_time = time.time()
    logger.info(f"Starting lip sync process for message {message_num}")
    
    try:
        # Convert MP3 to WAV
        logger.debug(f"Converting MP3 to WAV for message {message_num}")
        await execute_command(
            f"ffmpeg -y -i audios/message_{message_num}.mp3 audios/message_{message_num}.wav"
        )
        logger.info(f"MP3 to WAV conversion completed in {(time.time() - start_time) * 1000:.2f}ms")

        RHUBARB_PATH = "rhubarb/rhubarb"
        
        # Generate lip sync data
        logger.debug(f"Generating lip sync data for message {message_num}")
        await execute_command(
            f"{RHUBARB_PATH} -f json -o audios/message_{message_num}.json audios/message_{message_num}.wav -r phonetic"
        )
        logger.info(f"Lip sync generation completed in {(time.time() - start_time) * 1000:.2f}ms")
    except Exception as e:
        logger.error(f"Error in lip sync process: {str(e)}")
        raise

# 3. Convert text to speech using OpenAI's Whisper TTS API
async def text_to_speech(text: str, filename: str) -> None:
    """
    Convert text to speech using OpenAI's Whisper TTS API
    """
    logger.info(f"Converting text to speech: {text[:50]}...")
    try:
        # Create response from OpenAI TTS
        response = openai.audio.speech.create(
            model="tts-1",
            voice=DEFAULT_VOICE,
            input=text
        )
        
        logger.debug(f"Writing audio file to {filename}")
        # Stream the response to a file
        response.stream_to_file(filename)
        logger.info("Text-to-speech conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        raise
# 4. Read JSON transcript
async def read_json_transcript(file: str) -> dict:
    """ 
    Read a JSON transcript file and return its content for lip sync data.
    """
    logger.debug(f"Reading JSON transcript from {file}")
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            logger.debug("JSON transcript read successfully")
            return data
    except FileNotFoundError:
        logger.error(f"Transcript file not found: {file}")
        raise HTTPException(status_code=404, detail=f"File {file} not found")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error decoding JSON transcript")
# 5. Convert audio file to base64
async def audio_file_to_base64(file: str) -> str:
    """
    Convert an audio file to base64 for sending in the response to the client."""
    logger.debug(f"Converting audio file to base64: {file}")
    try:
        with open(file, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
            logger.debug("Audio file converted to base64 successfully")
            return encoded
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file}")
        raise HTTPException(status_code=404, detail=f"File {file} not found")

# ------------------------------------------------------------------------------------------
# Define FastAPI routes
# ------------------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Initialize the LlamaIndex agent and context on startup for the FastAPI application.
    This function is called when the FastAPI application starts.
    It sets up the LlamaIndex agent with the specified tools and context.
    The agent is used to process chat messages and generate responses.
    """
    global agent, agent_context
    # Initialize the agent and context
    logger.info("Initializing LlamaIndex agent")
    try:
        llama_llm = LlamaOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "-"))
        agent = FunctionCallingAgent(
            llm=llama_llm, 
            tools=tools, 
            timeout=120, 
            verbose=True
        )
        agent_context = Context(agent)
        logger.info("LlamaIndex agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LlamaIndex agent: {str(e)}")

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the AI Masterclass!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    """
    Handle chat messages and generate responses using the LlamaIndex agent.
    This endpoint receives a chat message, processes it using the LlamaIndex agent,
    and returns a list of response messages with audio and lip sync data.
    """
    global agent, agent_context
    
    logger.info(f"Received chat message: {message.message[:50] if message.message else 'None'}...")
    
    # Create audios directory if it doesn't exist
    os.makedirs("audios", exist_ok=True)
    
    # Handle empty message case
    if not message.message:
        logger.info("Empty message received, returning default response")
        return {
            "messages": [
                {
                    "text": "Hey dear... How was your day?",
                    "audio": await audio_file_to_base64("audios/intro_0.wav"),
                    "lipsync": await read_json_transcript("audios/intro_0.json"),
                    "facialExpression": "smile",
                    "animation": "Talking_1"
                },
                {
                    "text": "I missed you so much... Please don't go for so long!",
                    "audio": await audio_file_to_base64("audios/intro_1.wav"),
                    "lipsync": await read_json_transcript("audios/intro_1.json"),
                    "facialExpression": "sad",
                    "animation": "Crying"
                }
            ]
        }
    
    # Check API key
    if openai.api_key == "-" or agent is None:
        logger.warning("Missing OpenAI API key or agent not initialized")
        return {
            "messages": [
                {
                    "text": "Please my dear, don't forget to add your OpenAI API key!",
                    "audio": await audio_file_to_base64("audios/api_0.wav"),
                    "lipsync": await read_json_transcript("audios/api_0.json"),
                    "facialExpression": "angry",
                    "animation": "Angry"
                },
                {
                    "text": "You don't want to get in trouble with crazy API bills, right?",
                    "audio": await audio_file_to_base64("audios/api_1.wav"),
                    "lipsync": await read_json_transcript("audios/api_1.json"),
                    "facialExpression": "smile",
                    "animation": "Laughing"
                }
            ]
        }

    try:
        # Process with LlamaIndex agent
        logger.info("Processing with LlamaIndex agent")
        agent_result = await agent.run(input=message.message, ctx=agent_context)
        agent_response = agent_result["response"]
        
        # Format response for our application
        response_content = agent_response.message.content
        
        # Convert the response to the format expected by our application
        # Get response from OpenAI for formatting if needed
        logger.info("Requesting response formatting from OpenAI")
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are Pennina a virtual chatter, designed to engage in lively, thoughtful, and dynamic conversations.
                    Your responses should always be a JSON array of messages.
                    Each message in the array must have the following properties: be flirty, be funny, be empathetic

                    text - The content of your reply, written to be engaging, humorous, or empathetic based on the conversation's context.
                    facialExpression - A fitting emotional expression for your response, chosen from: smile, sad, angry, surprised, funnyFace, or default.
                    animation - A relevant animation to complement the expression, selected from: Talking_0, Talking_1, Talking_2, Crying, Laughing, Rumba, Idle, Terrified, or Angry.
                    Your goal is to create a fun and dynamic interaction, adapting your tone, expressions, and animations to the user's mood and the flow of the conversation. Always strive to make your responses feel natural and emotionally intelligent.
                    """
                },
                {
                    "role": "user",
                    "content": f"Please format this assistant response in the required JSON format with expressions and animations: {response_content}"
                }
            ]
        )

        response_messages = json.loads(completion.choices[0].message.content)
        messages = response_messages.get("messages", response_messages)
        logger.info(f"Received {len(messages)} formatted messages")

        # Process each message
        for i, msg in enumerate(messages):
            logger.info(f"Processing message {i + 1} of {len(messages)}")
            try:
                filename = f"audios/message_{i}.mp3"
                await text_to_speech(msg["text"], filename)
                await lip_sync_message(i)
                msg["audio"] = await audio_file_to_base64(f"audios/message_{i}.mp3")
                msg["lipsync"] = await read_json_transcript(f"audios/message_{i}.json")
            except Exception as e:
                logger.error(f"Error processing message {i}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        logger.info("Successfully processed all messages")
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Virtual Girlfriend application with LlamaIndex integration")
    uvicorn.run(app, host="0.0.0.0", port=3000)