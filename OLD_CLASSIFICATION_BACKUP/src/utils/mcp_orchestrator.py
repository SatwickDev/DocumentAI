"""
MCP Orchestrator - Manages communication between multiple MCP servers
Provides the missing contracts and inter-server communication
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .mcp_contracts import (
    MCPRequest, MCPResponse, MCPMethods, MCPErrors,
    validate_mcp_request, validate_mcp_response,
    create_mcp_error_response, create_mcp_success_response
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Configuration for an MCP server"""
    name: str
    script_path: str
    timeout: float = 60.0  # Increased timeout
    max_retries: int = 5   # Increased retries
    startup_delay: float = 5.0  # Added startup delay

class MCPClient:
    """Client for communicating with a single MCP server"""
    
    def __init__(self, server_config: ServerConfig):
        self.config = server_config
        self.process = None
        self.is_connected = False
        self.request_counter = 0
        self.ready_event = asyncio.Event()
        self._startup_lock = asyncio.Lock()  # Lock to prevent concurrent startup attempts
        
    async def _initialize_connection(self) -> bool:
        """Initialize the connection with proper checks and waiting"""
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Initialization attempt {attempt + 1}/{self.config.max_retries} for {self.config.name}")
                
                # Start the process
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable, ["-u", self.config.script_path],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for initialization
                start_time = time.time()
                while time.time() - start_time < self.config.timeout:
                    if self.process.returncode is not None:
                        error = await self.process.stderr.read()
                        logger.error(f"Process failed: {error.decode()}")
                        break
                        
                    try:
                        line = await asyncio.wait_for(
                            self.process.stdout.readline(),
                            timeout=1.0
                        )
                        if line:
                            message = line.decode().strip()
                            logger.info(f"{self.config.name}: {message}")
                            if "ready for connections" in message.lower():
                                self.is_connected = True
                                return True
                    except asyncio.TimeoutError:
                        continue
                        
                logger.warning(f"Timeout waiting for {self.config.name} initialization")
                
            except Exception as e:
                    logger.error(f"Initialization error for {self.config.name}: {e}")
                
            # Clean up failed attempt
            if self.process:
                try:
                    self.process.terminate()
                    await self.process.wait()
                except:
                    pass
                    
            await asyncio.sleep(1)
            
        logger.error(f"Failed to initialize {self.config.name} after {self.config.max_retries} attempts")
        return False
        
    async def connect(self) -> bool:
        """Connect to the MCP server with proper initialization handling"""
        if not await self._startup_lock.acquire():
            return False
            
        try:
            logger.info(f"🔄 Connecting to MCP server: {self.config.name}")
            
            # Clean up any existing process
            if self.process:
                try:
                    logger.info(f"Cleaning up existing process for {self.config.name}")
                    self.process.terminate()
                    await self.process.wait()
                except Exception as e:
                    logger.warning(f"Error cleaning up process: {e}")
                    
            # Initialize connection
            success = await self._initialize_connection()
            if success:
                self.ready_event.set()
                    logger.info(f"{self.config.name} initialized successfully")
            return success
            
        except Exception as e:
                logger.error(f"Connection error for {self.config.name}: {e}")
            return False
            
        finally:
            self._startup_lock.release()
            
            # Ensure script path exists
            script_path = os.path.abspath(self.config.script_path)
            if not os.path.isfile(script_path):
                    logger.error(f"Script not found at {script_path}")
                return False
            
            logger.info(f"Starting script: {script_path}")
            
            # Start process with unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONPATH'] = os.path.dirname(os.path.dirname(script_path))
            
            # Create process with proper error handling
            try:
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    script_path,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.path.dirname(script_path),
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
            except Exception as e:
                    logger.error(f"Failed to start server process: {e}")
                return False
            
            # Give process time to start
            await asyncio.sleep(0.5)
            
            # Check for startup errors
            if self.process.stderr:
                error = await self.process.stderr.readline()
                if error:
                    error_text = error.decode().strip()
                        logger.error(f"Server startup error for {self.config.name}: {error_text}")
                    return False
            
            # Test connection with retries
            test_request = MCPRequest(
                id=0,
                method=MCPMethods.TOOLS_LIST
            )
            
            max_attempts = 3
            for attempt in range(max_attempts):
                if attempt > 0:
                    logger.info(f"Connection attempt {attempt + 1}/{max_attempts}...")
                    await asyncio.sleep(attempt * 2)  # Exponential backoff
                
                try:
                    response = await asyncio.wait_for(
                        self._send_request(test_request),
                        timeout=self.config.timeout
                    )
                    if response and not response.error:
                        self.is_connected = True
                            logger.info(f"Connected to {self.config.name}")
                        return True
                except asyncio.TimeoutError:
                    logger.warning(f"Connection attempt {attempt + 1} timed out")
                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    continue
            
                logger.error(f"Failed to connect after {max_attempts} attempts")
            return False

        except Exception as e:
                logger.error(f"Connection error for {self.config.name}: {e}")
            return False

        finally:
            if not self.is_connected and self.process:
                try:
                    self.process.terminate()
                    await self.process.wait()
                except:
                    pass
                self.process = None
            
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
                logger.info(f"Disconnected from {self.config.name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {self.config.name}: {e}")
        
        self.is_connected = False
        self.process = None
    
    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send a request to the MCP server and get response"""
        try:
            # Check process is alive
            if not self.process or self.process.returncode is not None:
                raise Exception("Server process is not running")
            
            # Send request with error handling
            request_json = json.dumps(request.to_json()) + "\n"
            try:
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionError) as e:
                raise Exception(f"Failed to send request: {e}")
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"No response received after {self.config.timeout} seconds")
            
            if not response_line:
                raise Exception("Empty response received")
            
            response_text = response_line.decode().strip()
            
            # Parse and validate response
            try:
                response_dict = json.loads(response_text)
                validate_mcp_response(response_dict)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response: {e}")
            except Exception as e:
                raise Exception(f"Invalid response format: {e}")
            
            return MCPResponse.from_json(response_text)
            
        except Exception as e:
            logger.error(f"Request failed for {self.config.name}: {e}")
            return create_mcp_error_response(
                request.id,
                MCPErrors.INTERNAL_ERROR,
                str(e)
            )
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.is_connected:
            raise Exception(f"Not connected to {self.config.name}")
        
        self.request_counter += 1
        request = MCPRequest(
            id=self.request_counter,
            method=MCPMethods.TOOLS_CALL,
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        timeout = timeout or self.config.timeout
        
        try:
            response = await asyncio.wait_for(
                self._send_request(request),
                timeout=timeout
            )
            
            if response.error:
                raise Exception(f"Tool call failed: {response.error}")
            
            return response.result
            
        except asyncio.TimeoutError:
            raise Exception(f"Tool call timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Tool call failed for {self.config.name}.{tool_name}: {e}")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        if not self.is_connected:
            raise Exception(f"Not connected to {self.config.name}")
        
        self.request_counter += 1
        request = MCPRequest(
            id=self.request_counter,
            method=MCPMethods.TOOLS_LIST
        )
        
        try:
            response = await asyncio.wait_for(
                self._send_request(request),
                timeout=self.config.timeout
            )
            
            if response.error:
                raise Exception(f"Tools list failed: {response.error}")
            
            return response.result
            
        except asyncio.TimeoutError:
            raise Exception(f"Tools list timed out after {self.config.timeout}s")
        except Exception as e:
            logger.error(f"Tools list failed for {self.config.name}: {e}")
            raise

class MCPOrchestrator:
    """Orchestrates communication between multiple MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, ServerConfig] = {}
        self.clients: Dict[str, MCPClient] = {}
        self.is_initialized = False
    
    def register_server(self, server_config: ServerConfig):
        """Register an MCP server"""
        self.servers[server_config.name] = server_config
        logger.info(f"Registered MCP server: {server_config.name}")
    
    async def initialize(self) -> bool:
        """Initialize all registered servers"""
        if self.is_initialized:
            return True
        
        logger.info("Initializing MCP Orchestrator...")
        success_count = 0
        retries = 3
        
        try:
            for attempt in range(retries):
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{retries}...")
                    await asyncio.sleep(attempt * 2)  # Exponential backoff
                
                for server_name, server_config in self.servers.items():
                    if server_name in self.clients:  # Skip already connected servers
                        success_count += 1
                        continue
                    
                    try:
                        client = MCPClient(server_config)
                        connected = await client.connect()
                        
                        if connected:
                            self.clients[server_name] = client
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to initialize {server_name}: {e}")
                        if server_name in self.clients:
                            await self.clients[server_name].disconnect()
                            del self.clients[server_name]
                
                # Check if we have all servers connected
                if success_count == len(self.servers):
                    logger.info("✅ All servers initialized successfully")
                    self.is_initialized = True
                    return True
                
                # Continue retrying if we haven't connected all servers yet
                if attempt < retries - 1 and success_count < len(self.servers):
                    await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff
            
            # After all retries, check if we have at least one server
            self.is_initialized = success_count > 0
            if self.is_initialized:
                logger.warning(f"Partial initialization: {success_count}/{len(self.servers)} servers connected")
                return True
            else:
                logger.error("❌ No servers could be initialized")
                return False
        
        except Exception as e:
            logger.error(f"❌ Global initialization error: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all MCP clients"""
        logger.info("Shutting down MCP Orchestrator...")
        
        for client in self.clients.values():
            await client.disconnect()
        
        self.clients.clear()
        self.is_initialized = False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific server"""
        if not self.is_initialized:
            raise Exception("Orchestrator not initialized")
        
        if server_name not in self.clients:
            raise Exception(f"Server not available: {server_name}")
        
        client = self.clients[server_name]
        return await client.call_tool(tool_name, arguments)
    
    async def call_multiple_tools(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call multiple tools in parallel across different servers
        
        operations format:
        [
            {
                "server": "classification",
                "tool": "classify_document", 
                "arguments": {"file_path": "/path/to/doc.pdf"}
            }
        ]
        """
        if not self.is_initialized:
            raise Exception("Orchestrator not initialized")
        
        start_time = time.time()
        tasks = []
        task_info = []
        
        for op in operations:
            server_name = op["server"]
            tool_name = op["tool"]
            arguments = op.get("arguments", {})
            
            if server_name not in self.clients:
                logger.warning(f"Server not available: {server_name}")
                continue
            
            tasks.append(self.call_tool(server_name, tool_name, arguments))
            task_info.append({
                "server": server_name,
                "tool": tool_name
            })
        
        # Execute all tasks in parallel
        results = {}
        errors = {}
        
        if tasks:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(completed_tasks):
                info = task_info[i]
                server_name = info["server"]
                
                if isinstance(result, Exception):
                    logger.error(f"Task failed - {server_name}: {result}")
                    errors[server_name] = str(result)
                else:
                    results[server_name] = result
                    logger.info(f"Task completed - {server_name}")
        
        processing_time = time.time() - start_time
        
        return {
            "success": len(results) > 0,
            "results": results,
            "errors": errors,
            "tasks_requested": len(operations),
            "tasks_completed": len(results),
            "tasks_failed": len(errors),
            "processing_time_seconds": round(processing_time, 3),
            "timestamp": time.time()
        }
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers"""
        status = {
            "orchestrator": {
                "initialized": self.is_initialized,
                "servers_registered": len(self.servers),
                "clients_connected": len(self.clients)
            },
            "servers": {}
        }
        
        for server_name, client in self.clients.items():
            try:
                # Test connection by listing tools
                tools = await client.list_tools()
                status["servers"][server_name] = {
                    "status": "healthy",
                    "connected": client.is_connected,
                    "tools_count": len(tools),
                    "tools": [tool.get("name") for tool in tools[:3]]  # First 3 tools
                }
            except Exception as e:
                status["servers"][server_name] = {
                    "status": "unhealthy",
                    "connected": False,
                    "error": str(e)
                }
        
        return status
