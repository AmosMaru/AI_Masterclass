from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes import api_router
from app.logger import setup_logger

# Setup logger
logger = setup_logger()

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Virtual Character application with LlamaIndex integration")
    uvicorn.run(app, host="0.0.0.0", port=3000)
    # uvicorn main:app --host 0.0.0.0 --port 3000 --reload  