#!/usr/bin/env python3
"""
MCP Server Runner
Starts MCP servers and handles communication
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def start_mcp_server(server_module: str, name: str):
    """Start an MCP server"""
    try:
        module = __import__(f"src.mcp_servers.{server_module}", fromlist=['main'])
        logger.info(f"Starting {name}...")
        await module.main()
    except Exception as e:
        logger.error(f"Failed to start {name}: {e}")

async def main():
    """Start all MCP servers"""
    servers = [
        ('classification_mcp_server', 'Classification MCP Server'),
        ('quality_mcp_server', 'Quality MCP Server')
    ]
    
    tasks = [start_mcp_server(module, name) for module, name in servers]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down MCP servers...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
