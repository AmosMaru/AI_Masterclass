import os
import json
from app.agent import FunctionCallingAgent
from app.logger import setup_logger
from app.schema import ChatResponse, Message
from app.tools import get_time, get_weather, remember_fact, search_serper, send_sms
from utils.helper import (
    audio_file_to_base64,
    lip_sync_message,
    read_json_transcript,
    text_to_speech,
)
from fastapi import HTTPException
from fastapi import APIRouter
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.tools import FunctionTool

load_dotenv()
logger = setup_logger()

api_router = APIRouter()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# Global agent and context objects
agent = None
agent_context = None

# Initialize tools and agent
tools = [
    FunctionTool.from_defaults(get_weather),
    FunctionTool.from_defaults(get_time),
    FunctionTool.from_defaults(remember_fact),
    FunctionTool.from_defaults(send_sms),
    FunctionTool.from_defaults(search_serper)
]

print(dir(tools[0]))


tool_descriptions = "\n".join(
    f"- {tool.metadata.name}: {tool.metadata.description or 'No description'}"
    for tool in tools
)


@api_router.on_event("startup")
async def startup_event():
    global agent, agent_context
    # Initialize the agent and context
    logger.info("Initializing LlamaIndex agent")
    try:
        llama_llm = LlamaOpenAI(
            model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "-")
        )
        agent = FunctionCallingAgent(
            llm=llama_llm, tools=tools, timeout=120, verbose=True
        )
        agent_context = Context(agent)
        logger.info("LlamaIndex agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LlamaIndex agent: {str(e)}")


@api_router.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello World!"}


@api_router.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    global agent, agent_context

    logger.info(
        f"Received chat message: {message.message[:50] if message.message else 'None'}..."
    )

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
                    "animation": "Talking_1",
                },
                {
                    "text": "I missed you so much... Please don't go for so long!",
                    "audio": await audio_file_to_base64("audios/intro_1.wav"),
                    "lipsync": await read_json_transcript("audios/intro_1.json"),
                    "facialExpression": "sad",
                    "animation": "Crying",
                },
            ]
        }

    # Check API key
    if client.api_key == "-" or agent is None:
        logger.warning("Missing OpenAI API key or agent not initialized")
        return {
            "messages": [
                {
                    "text": "Please my dear, don't forget to add your OpenAI API key!",
                    "audio": await audio_file_to_base64("audios/api_0.wav"),
                    "lipsync": await read_json_transcript("audios/api_0.json"),
                    "facialExpression": "angry",
                    "animation": "Angry",
                },
                {
                    "text": "You don't want to get in trouble with crazy API bills, right?",
                    "audio": await audio_file_to_base64("audios/api_1.wav"),
                    "lipsync": await read_json_transcript("audios/api_1.json"),
                    "facialExpression": "smile",
                    "animation": "Laughing",
                },
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
        completion = client.chat.completions.create(
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

                    You have access to the following tools:
                    {tool_descriptions}

                    When you need external information (like a web search or current weather), you can call these tools.
                    Always strive to be flirty, funny, and empathetic in your responses.

                    Your responses must always be a JSON array of messages.
                    Each message must have these properties:
                    - text: The content of your reply, written to be engaging, humorous, or empathetic based on the conversation's context.
                    - facialExpression: A fitting emotional expression for your response, chosen from: smile, sad, angry, surprised, funnyFace, or default.
                    - animation: A relevant animation to complement the expression, selected from: Talking_0, Talking_1, Talking_2, Crying, Laughing, Rumba, Idle, Terrified, or Angry.

                    Adapt your tone, expressions, and animations to the user's mood and the flow of conversation. Be emotionally intelligent, natural, and entertaining.
                    """,
                },
                {
                    "role": "user",
                    "content": f"Please format this assistant response in the required JSON format with expressions and animations: {response_content}. ",
                },
            ],
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
