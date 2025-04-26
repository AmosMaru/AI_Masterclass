import asyncio
from app.logger import setup_logger
from openai import OpenAI
from fastapi import HTTPException
import os
import time
import json
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize logger
logger = setup_logger()
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DEFAULT_VOICE = "alloy"


async def execute_command(command: str) -> str:
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

async def lip_sync_message(message_num: int) -> None:
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

async def text_to_speech(text: str, filename: str) -> None:
    """
    Convert text to speech using OpenAI's Whisper TTS API
    """
    logger.info(f"Converting text to speech: {text[:50]}...")
    try:
        # Create response from OpenAI TTS
        response = client.audio.speech.create(
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

async def read_json_transcript(file: str) -> dict:
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

async def audio_file_to_base64(file: str) -> str:
    logger.debug(f"Converting audio file to base64: {file}")
    try:
        with open(file, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
            logger.debug("Audio file converted to base64 successfully")
            return encoded
    except FileNotFoundError:
        logger.error(f"Audio file not found: {file}")
        raise HTTPException(status_code=404, detail=f"File {file} not found")
