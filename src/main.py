"""
Smart Shades Agent - Main Application Entry Point
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.smart_shades_agent_v2 import SmartShadesAgentV2
from api import root, rooms, schedules

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent

    # Startup
    logger.info("Starting Smart Shades Agent V2...")
    agent = SmartShadesAgentV2()
    await agent.initialize()

    # Inject agent into API modules
    rooms.set_agent(agent)
    schedules.set_agent(agent)

    logger.info("Smart Shades Agent V2 initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Smart Shades Agent V2...")
    if agent:
        await agent.shutdown()
    logger.info("Smart Shades Agent V2 shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Smart Shades Agent API",
    description="""
    Intelligent shade control system using LangChain and solar intelligence.
    
    ## Features
    * Natural language shade control
    * Solar-aware automation  
    * Scheduling capabilities
    * Room-based management
    * Apple Shortcuts integration
    
    ## Getting Started
    1. Use `/rooms` to see available rooms
    2. Control shades with `/rooms/{room}/control`
    3. Create schedules with `/rooms/{room}/schedules`
    4. View solar data with `/rooms/{room}/solar`
    """,
    version="2.0.0",
    contact={"name": "Smart Shades Support", "email": "support@example.com"},
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(root.router)
app.include_router(rooms.router)
app.include_router(schedules.router)


async def main():
    """Main application function"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Starting Smart Shades Agent API on {host}:{port}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
