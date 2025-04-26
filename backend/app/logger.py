import os
import sys
import logging
from logging.handlers import RotatingFileHandler

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
