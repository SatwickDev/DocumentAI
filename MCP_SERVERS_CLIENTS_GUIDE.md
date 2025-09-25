# MCP Servers and Clients - Complete Technical Guide

## Table of Contents
1. [MCP Server Architecture](#mcp-server-architecture)
2. [MCP Client Implementation](#mcp-client-implementation)
3. [MCP Orchestrator](#mcp-orchestrator)
4. [Communication Protocols](#communication-protocols)
5. [Error Handling and Recovery](#error-handling-and-recovery)
6. [Performance Optimization](#performance-optimization)
7. [Security Implementation](#security-implementation)
8. [Testing Strategies](#testing-strategies)

## MCP Server Architecture

### Classification MCP Server

#### Server Structure
```python
# src/mcp_servers/classification_mcp_server.py

class ClassificationMCPServer:
    """
    MCP Server for document classification operations
    Implements JSON-RPC 2.0 protocol for standardized communication
    """
    
    def __init__(self):
        self.server_name = "classification_server"
        self.version = "2.0.0"
        self.capabilities = [
            "document_classification",
            "text_extraction", 
            "keyword_analysis",
            "confidence_scoring"
        ]
        self.methods = {
            "classify_document": self.classify_document,
            "extract_text": self.extract_text,
            "get_categories": self.get_categories,
            "update_model": self.update_model,
            "get_statistics": self.get_statistics,
            "health_check": self.health_check
        }
        
        # Initialize core components
        self.document_classifier = DocumentClassifier()
        self.text_extractor = TextExtractor()
        self.confidence_calculator = ConfidenceCalculator()
        
    async def classify_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a document and return category with confidence score
        
        Args:
            params: {
                "request_id": str,
                "file_path": str,
                "filename": str,
                "classification_type": str,
                "options": dict
            }
        
        Returns:
            {
                "category": str,
                "confidence": float,
                "keywords": List[str],
                "processing_method": str,
                "metadata": dict
            }
        """
        try:
            start_time = time.time()
            
            # Validate input parameters
            await self._validate_classification_params(params)
            
            # Extract text from document
            text_content = await self.text_extractor.extract(
                file_path=params["file_path"],
                file_type=Path(params["filename"]).suffix
            )
            
            # Preprocess text
            processed_text = await self.document_classifier.preprocess_text(text_content)
            
            # Perform classification
            classification_result = await self.document_classifier.classify(
                text=processed_text,
                classification_type=params.get("classification_type", "automatic"),
                options=params.get("options", {})
            )
            
            # Calculate confidence score
            confidence_score = await self.confidence_calculator.calculate(
                text=processed_text,
                predicted_category=classification_result.category,
                model_output=classification_result.raw_output
            )
            
            # Extract keywords
            keywords = await self.document_classifier.extract_keywords(
                text=processed_text,
                category=classification_result.category
            )
            
            processing_time = time.time() - start_time
            
            return {
                "category": classification_result.category,
                "confidence": confidence_score,
                "keywords": keywords,
                "processing_method": classification_result.method,
                "processing_time": processing_time,
                "metadata": {
                    "text_length": len(text_content),
                    "processed_text_length": len(processed_text),
                    "model_version": self.document_classifier.model_version,
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Classification error for {params.get('filename', 'unknown')}: {e}")
            raise MCPServerError(f"Classification failed: {str(e)}")
    
    async def extract_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text content from document without classification"""
        try:
            text_content = await self.text_extractor.extract(
                file_path=params["file_path"],
                file_type=Path(params["filename"]).suffix
            )
            
            return {
                "text_content": text_content,
                "text_length": len(text_content),
                "extraction_method": self.text_extractor.last_method_used
            }
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise MCPServerError(f"Text extraction failed: {str(e)}")
    
    async def get_categories(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get list of available classification categories"""
        return {
            "categories": self.document_classifier.available_categories,
            "total_count": len(self.document_classifier.available_categories),
            "model_version": self.document_classifier.model_version
        }
    
    async def get_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get server performance statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "successful_classifications": self.stats.successful_classifications,
            "failed_classifications": self.stats.failed_classifications,
            "average_processing_time": self.stats.average_processing_time,
            "uptime": time.time() - self.start_time,
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent()
        }
```

### Quality MCP Server

#### Server Implementation
```python
# src/mcp_servers/quality_mcp_server.py

class QualityMCPServer:
    """
    MCP Server for document quality analysis operations
    Provides comprehensive quality assessment and recommendations
    """
    
    def __init__(self):
        self.server_name = "quality_server"
        self.version = "2.0.0"
        self.capabilities = [
            "quality_analysis",
            "readability_assessment",
            "structure_validation",
            "issue_detection",
            "recommendation_generation"
        ]
        self.methods = {
            "analyze_quality": self.analyze_quality,
            "check_readability": self.check_readability,
            "validate_structure": self.validate_structure,
            "detect_issues": self.detect_issues,
            "generate_recommendations": self.generate_recommendations,
            "get_quality_metrics": self.get_quality_metrics,
            "health_check": self.health_check
        }
        
        # Initialize quality analysis components
        self.quality_analyzer = QualityAnalyzer()
        self.readability_checker = ReadabilityChecker()
        self.structure_validator = StructureValidator()
        self.issue_detector = IssueDetector()
        self.recommendation_engine = RecommendationEngine()
        
    async def analyze_quality(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive quality analysis of a document
        
        Args:
            params: {
                "request_id": str,
                "file_path": str,
                "filename": str,
                "analysis_type": str,
                "options": dict
            }
        
        Returns:
            {
                "overall_score": float,
                "readability_score": float,
                "structure_score": float,
                "content_quality_score": float,
                "issues": List[str],
                "recommendations": List[str],
                "metrics": dict
            }
        """
        try:
            start_time = time.time()
            
            # Validate parameters
            await self._validate_quality_params(params)
            
            # Extract document content
            content = await self.quality_analyzer.extract_content(
                file_path=params["file_path"],
                file_type=Path(params["filename"]).suffix
            )
            
            # Perform readability analysis
            readability_result = await self.readability_checker.analyze(
                text=content.text,
                options=params.get("options", {})
            )
            
            # Validate document structure
            structure_result = await self.structure_validator.validate(
                content=content,
                document_type=content.detected_type
            )
            
            # Analyze content quality
            content_quality_result = await self.quality_analyzer.analyze_content_quality(
                content=content,
                readability=readability_result
            )
            
            # Detect issues
            issues = await self.issue_detector.detect_all_issues(
                content=content,
                readability=readability_result,
                structure=structure_result,
                content_quality=content_quality_result
            )
            
            # Generate recommendations
            recommendations = await self.recommendation_engine.generate(
                issues=issues,
                content=content,
                quality_scores={
                    "readability": readability_result.score,
                    "structure": structure_result.score,
                    "content_quality": content_quality_result.score
                }
            )
            
            # Calculate overall score
            overall_score = await self._calculate_overall_score(
                readability_result.score,
                structure_result.score,
                content_quality_result.score
            )
            
            processing_time = time.time() - start_time
            
            return {
                "overall_score": overall_score,
                "readability_score": readability_result.score,
                "structure_score": structure_result.score,
                "content_quality_score": content_quality_result.score,
                "issues": [issue.description for issue in issues],
                "recommendations": [rec.description for rec in recommendations],
                "processing_time": processing_time,
                "metrics": {
                    "word_count": content.word_count,
                    "sentence_count": content.sentence_count,
                    "paragraph_count": content.paragraph_count,
                    "readability_metrics": readability_result.detailed_metrics,
                    "structure_metrics": structure_result.detailed_metrics,
                    "content_metrics": content_quality_result.detailed_metrics,
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Quality analysis error for {params.get('filename', 'unknown')}: {e}")
            raise MCPServerError(f"Quality analysis failed: {str(e)}")
```

## MCP Client Implementation

### MCP Connection Manager
```python
# src/utils/mcp_client.py

class MCPClient:
    """
    Client for communicating with MCP servers
    Handles connection management, request/response processing, and error handling
    """
    
    def __init__(self, server_config: ServerConfig):
        self.server_config = server_config
        self.connection = None
        self.request_id_counter = 0
        self.pending_requests = {}
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        try:
            # Start server process if not running
            if not await self._is_server_running():
                await self._start_server()
            
            # Establish connection (could be TCP, HTTP, or process communication)
            self.connection = await self._establish_connection()
            
            # Perform handshake
            handshake_result = await self._perform_handshake()
            
            if handshake_result:
                self.is_connected = True
                logger.info(f"Connected to MCP server: {self.server_config.name}")
                return True
            else:
                logger.error(f"Handshake failed with MCP server: {self.server_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_config.name}: {e}")
            return False
    
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server and return response"""
        if not self.is_connected:
            await self.connect()
        
        # Generate unique request ID
        request_id = self._generate_request_id()
        
        # Create MCP request
        mcp_request = MCPRequest(
            id=request_id,
            method=method,
            params=params
        )
        
        try:
            # Send request
            await self._send_raw_request(mcp_request)
            
            # Wait for response with timeout
            response = await self._wait_for_response(request_id, timeout=30)
            
            if response.error:
                raise MCPServerError(f"Server error: {response.error}")
            
            return response.result
            
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Request {request_id} timed out")
        except Exception as e:
            raise MCPCommunicationError(f"Communication error: {str(e)}")
    
    async def _send_raw_request(self, request: MCPRequest):
        """Send raw MCP request over connection"""
        request_json = request.to_json()
        
        if self.server_config.communication_type == "http":
            # HTTP-based communication
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_config.endpoint}/rpc",
                    data=request_json,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    response_data = await response.json()
                    return MCPResponse(**response_data)
        
        elif self.server_config.communication_type == "tcp":
            # TCP socket communication
            reader, writer = await asyncio.open_connection(
                self.server_config.host, 
                self.server_config.port
            )
            
            writer.write(request_json.encode() + b'\n')
            await writer.drain()
            
            response_data = await reader.readline()
            writer.close()
            await writer.wait_closed()
            
            return MCPResponse.from_json(response_data.decode())
        
        elif self.server_config.communication_type == "process":
            # Process-based communication (stdin/stdout)
            process = self.connection
            process.stdin.write(request_json.encode() + b'\n')
            await process.stdin.drain()
            
            response_data = await process.stdout.readline()
            return MCPResponse.from_json(response_data.decode())
```

### Connection Pool Management
```python
# src/utils/mcp_connection_pool.py

class MCPConnectionPool:
    """
    Manages a pool of MCP client connections for load balancing and failover
    """
    
    def __init__(self, server_configs: List[ServerConfig], pool_size: int = 5):
        self.server_configs = server_configs
        self.pool_size = pool_size
        self.connection_pools = {}
        self.load_balancer = RoundRobinLoadBalancer()
        
    async def initialize_pools(self):
        """Initialize connection pools for each server"""
        for config in self.server_configs:
            pool = []
            for _ in range(self.pool_size):
                client = MCPClient(config)
                if await client.connect():
                    pool.append(client)
            
            self.connection_pools[config.name] = pool
            logger.info(f"Initialized pool for {config.name} with {len(pool)} connections")
    
    async def get_client(self, server_name: str) -> MCPClient:
        """Get an available client from the pool"""
        pool = self.connection_pools.get(server_name)
        if not pool:
            raise MCPServerNotFoundError(f"No pool found for server: {server_name}")
        
        # Get healthy client from pool
        for client in pool:
            if client.is_connected and await self._health_check(client):
                return client
        
        # No healthy clients, try to create new one
        config = next((c for c in self.server_configs if c.name == server_name), None)
        if config:
            new_client = MCPClient(config)
            if await new_client.connect():
                pool.append(new_client)
                return new_client
        
        raise MCPConnectionError(f"No healthy connections available for {server_name}")
    
    async def return_client(self, client: MCPClient):
        """Return client to pool after use"""
        # Health check before returning to pool
        if await self._health_check(client):
            # Client is healthy, keep in pool
            pass
        else:
            # Client is unhealthy, remove from pool and create new one
            server_name = client.server_config.name
            pool = self.connection_pools[server_name]
            if client in pool:
                pool.remove(client)
                await client.disconnect()
            
            # Try to create replacement
            new_client = MCPClient(client.server_config)
            if await new_client.connect():
                pool.append(new_client)
```

## MCP Orchestrator

### Orchestrator Core Implementation
```python
# src/utils/mcp_orchestrator.py

class MCPOrchestrator:
    """
    Central orchestrator for managing multiple MCP servers and routing requests
    Provides high-level interface for microservices to interact with MCP servers
    """
    
    def __init__(self):
        self.servers = {}
        self.connection_pools = {}
        self.health_monitor = HealthMonitor()
        self.metrics_collector = MetricsCollector()
        self.circuit_breakers = {}
        
    async def register_server(self, config: ServerConfig):
        """Register a new MCP server with the orchestrator"""
        try:
            # Store server configuration
            self.servers[config.name] = config
            
            # Initialize connection pool
            pool = MCPConnectionPool([config], pool_size=config.pool_size or 3)
            await pool.initialize_pools()
            self.connection_pools[config.name] = pool
            
            # Initialize circuit breaker
            self.circuit_breakers[config.name] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30,
                expected_exception=MCPServerError
            )
            
            # Start health monitoring
            await self.health_monitor.add_server(config)
            
            logger.info(f"Registered MCP server: {config.name}")
            
        except Exception as e:
            logger.error(f"Failed to register server {config.name}: {e}")
            raise
    
    async def send_request(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to specified MCP server with fault tolerance"""
        
        # Check if server is registered
        if server_name not in self.servers:
            raise MCPServerNotFoundError(f"Server not found: {server_name}")
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers[server_name]
        if circuit_breaker.is_open():
            raise MCPServiceUnavailableError(f"Circuit breaker open for {server_name}")
        
        start_time = time.time()
        
        try:
            # Get client from connection pool
            pool = self.connection_pools[server_name]
            client = await pool.get_client(server_name)
            
            try:
                # Send request through circuit breaker
                with circuit_breaker:
                    result = await client.send_request(method, params)
                
                # Record successful request
                await self.metrics_collector.record_success(
                    server_name=server_name,
                    method=method,
                    response_time=time.time() - start_time
                )
                
                return result
                
            finally:
                # Return client to pool
                await pool.return_client(client)
        
        except Exception as e:
            # Record failed request
            await self.metrics_collector.record_failure(
                server_name=server_name,
                method=method,
                error=str(e),
                response_time=time.time() - start_time
            )
            
            logger.error(f"Request failed to {server_name}.{method}: {e}")
            raise
    
    async def broadcast_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Send request to all registered servers"""
        results = {}
        
        async def send_to_server(server_name: str):
            try:
                result = await self.send_request(server_name, method, params)
                results[server_name] = {"success": True, "result": result}
            except Exception as e:
                results[server_name] = {"success": False, "error": str(e)}
        
        # Send requests concurrently
        tasks = [send_to_server(name) for name in self.servers.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def get_server_status(self, server_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific server"""
        if server_name not in self.servers:
            return {"status": "not_found"}
        
        try:
            # Health check
            health_status = await self.health_monitor.check_server(server_name)
            
            # Get metrics
            metrics = await self.metrics_collector.get_server_metrics(server_name)
            
            # Circuit breaker status
            circuit_breaker = self.circuit_breakers[server_name]
            
            return {
                "status": "healthy" if health_status else "unhealthy",
                "health_details": health_status,
                "metrics": metrics,
                "circuit_breaker": {
                    "state": circuit_breaker.state,
                    "failure_count": circuit_breaker.failure_count,
                    "last_failure_time": circuit_breaker.last_failure_time
                },
                "connection_pool": {
                    "total_connections": len(self.connection_pools[server_name].connection_pools[server_name]),
                    "active_connections": await self._count_active_connections(server_name)
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

### Health Monitoring System
```python
# src/utils/health_monitor.py

class HealthMonitor:
    """
    Continuous health monitoring for MCP servers
    Provides proactive detection of server issues and automatic recovery
    """
    
    def __init__(self):
        self.monitored_servers = {}
        self.health_history = {}
        self.alert_manager = AlertManager()
        self.running = False
        
    async def add_server(self, config: ServerConfig):
        """Add server to health monitoring"""
        self.monitored_servers[config.name] = {
            "config": config,
            "last_check": None,
            "consecutive_failures": 0,
            "status": "unknown"
        }
        
        self.health_history[config.name] = deque(maxlen=100)  # Keep last 100 checks
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.running = True
        
        while self.running:
            try:
                # Check all servers
                for server_name in self.monitored_servers.keys():
                    await self._check_server_health(server_name)
                
                # Wait before next check cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)  # Short wait on error
    
    async def _check_server_health(self, server_name: str):
        """Perform health check for a specific server"""
        server_info = self.monitored_servers[server_name]
        config = server_info["config"]
        
        start_time = time.time()
        health_result = {
            "timestamp": start_time,
            "server_name": server_name,
            "checks": {}
        }
        
        try:
            # Basic connectivity check
            connectivity_ok = await self._check_connectivity(config)
            health_result["checks"]["connectivity"] = connectivity_ok
            
            # Health endpoint check
            if connectivity_ok:
                health_endpoint_ok = await self._check_health_endpoint(config)
                health_result["checks"]["health_endpoint"] = health_endpoint_ok
                
                # Response time check
                response_time = time.time() - start_time
                health_result["checks"]["response_time"] = {
                    "value": response_time,
                    "ok": response_time < 5.0  # 5 second threshold
                }
                
                # Resource usage check (if available)
                resource_usage = await self._check_resource_usage(config)
                health_result["checks"]["resource_usage"] = resource_usage
            
            # Determine overall health
            all_checks_ok = all(
                check.get("ok", True) if isinstance(check, dict) else check
                for check in health_result["checks"].values()
            )
            
            if all_checks_ok:
                server_info["status"] = "healthy"
                server_info["consecutive_failures"] = 0
            else:
                server_info["status"] = "unhealthy"
                server_info["consecutive_failures"] += 1
                
                # Alert if consecutive failures exceed threshold
                if server_info["consecutive_failures"] >= 3:
                    await self.alert_manager.send_alert(
                        severity="critical",
                        message=f"MCP Server {server_name} is unhealthy",
                        details=health_result
                    )
            
            health_result["overall_status"] = server_info["status"]
            server_info["last_check"] = time.time()
            
        except Exception as e:
            health_result["error"] = str(e)
            health_result["overall_status"] = "error"
            server_info["status"] = "error"
            server_info["consecutive_failures"] += 1
            
            logger.error(f"Health check failed for {server_name}: {e}")
        
        # Store health history
        self.health_history[server_name].append(health_result)
    
    async def _check_connectivity(self, config: ServerConfig) -> bool:
        """Check basic connectivity to server"""
        try:
            if config.communication_type == "http":
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{config.endpoint}/health", timeout=5) as response:
                        return response.status < 500
            
            elif config.communication_type == "tcp":
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(config.host, config.port),
                    timeout=5
                )
                writer.close()
                await writer.wait_closed()
                return True
            
            elif config.communication_type == "process":
                # Check if process is running
                return config.process and config.process.poll() is None
            
            return False
            
        except Exception:
            return False
```

## Communication Protocols

### JSON-RPC 2.0 Implementation
```python
# src/utils/json_rpc.py

class JSONRPCProtocol:
    """
    JSON-RPC 2.0 protocol implementation for MCP communication
    Handles request/response serialization and validation
    """
    
    @staticmethod
    def create_request(method: str, params: Any = None, request_id: Union[str, int] = None) -> str:
        """Create JSON-RPC 2.0 request"""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id or str(uuid.uuid4())
        }
        
        if params is not None:
            request["params"] = params
        
        return json.dumps(request)
    
    @staticmethod
    def create_response(result: Any = None, error: Dict = None, request_id: Union[str, int] = None) -> str:
        """Create JSON-RPC 2.0 response"""
        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        
        if error:
            response["error"] = error
        else:
            response["result"] = result
        
        return json.dumps(response)
    
    @staticmethod
    def parse_request(json_str: str) -> Dict[str, Any]:
        """Parse and validate JSON-RPC 2.0 request"""
        try:
            data = json.loads(json_str)
            
            # Validate required fields
            if data.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid Request", "jsonrpc must be '2.0'")
            
            if "method" not in data:
                raise JSONRPCError(-32600, "Invalid Request", "method is required")
            
            if not isinstance(data["method"], str):
                raise JSONRPCError(-32600, "Invalid Request", "method must be string")
            
            return data
            
        except json.JSONDecodeError as e:
            raise JSONRPCError(-32700, "Parse error", str(e))
        except Exception as e:
            raise JSONRPCError(-32600, "Invalid Request", str(e))
    
    @staticmethod
    def parse_response(json_str: str) -> Dict[str, Any]:
        """Parse and validate JSON-RPC 2.0 response"""
        try:
            data = json.loads(json_str)
            
            # Validate required fields
            if data.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid Response", "jsonrpc must be '2.0'")
            
            if "id" not in data:
                raise JSONRPCError(-32600, "Invalid Response", "id is required")
            
            # Must have either result or error, but not both
            has_result = "result" in data
            has_error = "error" in data
            
            if has_result and has_error:
                raise JSONRPCError(-32600, "Invalid Response", "response cannot have both result and error")
            
            if not has_result and not has_error:
                raise JSONRPCError(-32600, "Invalid Response", "response must have either result or error")
            
            return data
            
        except json.JSONDecodeError as e:
            raise JSONRPCError(-32700, "Parse error", str(e))
        except Exception as e:
            raise JSONRPCError(-32600, "Invalid Response", str(e))
```

### Protocol Error Handling
```python
# src/utils/mcp_errors.py

class MCPError(Exception):
    """Base exception for MCP-related errors"""
    pass

class JSONRPCError(MCPError):
    """JSON-RPC 2.0 standard errors"""
    
    # Standard error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC Error {code}: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to JSON-RPC error object"""
        error = {
            "code": self.code,
            "message": self.message
        }
        
        if self.data is not None:
            error["data"] = self.data
        
        return error

class MCPServerError(MCPError):
    """MCP server processing errors"""
    pass

class MCPConnectionError(MCPError):
    """MCP connection errors"""
    pass

class MCPTimeoutError(MCPError):
    """MCP request timeout errors"""
    pass

class MCPServerNotFoundError(MCPError):
    """MCP server not found errors"""
    pass

class MCPServiceUnavailableError(MCPError):
    """MCP service unavailable errors"""
    pass
```

## Error Handling and Recovery

### Circuit Breaker Pattern
```python
# src/utils/circuit_breaker.py

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for MCP servers
    Prevents cascade failures and provides automatic recovery
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "open":
            # Check if we should transition to half-open
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.is_open():
            raise MCPServiceUnavailableError("Circuit breaker is open")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type and issubclass(exc_type, self.expected_exception):
            # Failure occurred
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        elif exc_type is None and self.state == "half-open":
            # Success in half-open state, close the circuit
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful request")
        
        return False  # Don't suppress exceptions
```

### Retry Logic with Exponential Backoff
```python
# src/utils/retry_handler.py

class RetryHandler:
    """
    Handles request retries with exponential backoff and jitter
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except (MCPConnectionError, MCPTimeoutError) as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break
            
            except MCPServerError as e:
                # Don't retry server errors
                logger.error(f"Server error, not retrying: {e}")
                raise
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay += jitter
        
        return max(0, delay)
```

## Performance Optimization

### Connection Pooling Strategy
```python
# src/utils/performance_optimizer.py

class PerformanceOptimizer:
    """
    Performance optimization strategies for MCP communication
    """
    
    def __init__(self):
        self.request_cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute TTL
        self.compression_enabled = True
        self.batch_processor = BatchProcessor()
        
    async def optimize_request(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization strategies to request"""
        
        # 1. Check cache first
        cache_key = self._generate_cache_key(server_name, method, params)
        cached_result = self.request_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_result
        
        # 2. Check if request can be batched
        if self._is_batchable(method):
            result = await self.batch_processor.add_request(server_name, method, params)
        else:
            # 3. Apply compression if beneficial
            if self.compression_enabled and self._should_compress(params):
                params = await self._compress_params(params)
            
            # 4. Execute request normally
            result = await self._execute_request(server_name, method, params)
        
        # 5. Cache result if cacheable
        if self._is_cacheable(method, result):
            self.request_cache[cache_key] = result
        
        return result
    
    def _generate_cache_key(self, server_name: str, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        # Create deterministic hash of parameters
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{server_name}:{method}:{params_hash}"
    
    def _is_cacheable(self, method: str, result: Dict[str, Any]) -> bool:
        """Determine if result should be cached"""
        # Cache read-only operations
        readonly_methods = ['get_categories', 'get_statistics', 'health_check']
        return method in readonly_methods
    
    def _is_batchable(self, method: str) -> bool:
        """Determine if request can be batched"""
        batchable_methods = ['classify_document', 'analyze_quality']
        return method in batchable_methods
```

### Batch Processing Implementation
```python
# src/utils/batch_processor.py

class BatchProcessor:
    """
    Batches multiple requests together for improved throughput
    """
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_batches = {}
        
    async def add_request(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add request to batch processing queue"""
        
        batch_key = f"{server_name}:{method}"
        
        if batch_key not in self.pending_batches:
            self.pending_batches[batch_key] = {
                "requests": [],
                "futures": [],
                "timer": None
            }
        
        batch = self.pending_batches[batch_key]
        
        # Create future for this request
        future = asyncio.Future()
        
        # Add to batch
        batch["requests"].append(params)
        batch["futures"].append(future)
        
        # Start timer if this is the first request in batch
        if len(batch["requests"]) == 1:
            batch["timer"] = asyncio.create_task(
                self._batch_timeout_handler(batch_key)
            )
        
        # Execute batch if it's full
        if len(batch["requests"]) >= self.batch_size:
            await self._execute_batch(batch_key)
        
        # Wait for result
        return await future
    
    async def _execute_batch(self, batch_key: str):
        """Execute a batch of requests"""
        batch = self.pending_batches.pop(batch_key, None)
        if not batch:
            return
        
        # Cancel timeout timer
        if batch["timer"]:
            batch["timer"].cancel()
        
        server_name, method = batch_key.split(":", 1)
        
        try:
            # Send batch request
            batch_params = {
                "batch_requests": batch["requests"]
            }
            
            batch_result = await self._send_batch_request(server_name, f"batch_{method}", batch_params)
            
            # Distribute results to individual futures
            results = batch_result.get("results", [])
            for i, future in enumerate(batch["futures"]):
                if i < len(results):
                    future.set_result(results[i])
                else:
                    future.set_exception(MCPServerError("Batch result missing"))
        
        except Exception as e:
            # Set exception for all futures
            for future in batch["futures"]:
                future.set_exception(e)
    
    async def _batch_timeout_handler(self, batch_key: str):
        """Handle batch timeout"""
        await asyncio.sleep(self.batch_timeout)
        await self._execute_batch(batch_key)
```

## Security Implementation

### Authentication and Authorization
```python
# src/utils/mcp_security.py

class MCPSecurityManager:
    """
    Security management for MCP communication
    Handles authentication, authorization, and encryption
    """
    
    def __init__(self):
        self.api_keys = {}
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key")
        self.encryption_enabled = True
        
    async def authenticate_request(self, request_headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Authenticate incoming request"""
        
        # Check for API key
        api_key = request_headers.get("X-API-Key")
        if api_key:
            return await self._validate_api_key(api_key)
        
        # Check for JWT token
        auth_header = request_headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return await self._validate_jwt_token(token)
        
        return None
    
    async def _validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise SecurityError("Token expired")
            
            return payload
            
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {e}")
    
    async def authorize_request(self, user_info: Dict[str, Any], method: str, params: Dict[str, Any]) -> bool:
        """Authorize request based on user permissions"""
        
        user_role = user_info.get("role", "guest")
        user_permissions = user_info.get("permissions", [])
        
        # Define method permissions
        method_permissions = {
            "classify_document": ["read", "process"],
            "analyze_quality": ["read", "process"],
            "get_statistics": ["read", "admin"],
            "update_model": ["admin"],
            "health_check": ["read"]
        }
        
        required_permissions = method_permissions.get(method, ["admin"])
        
        # Check if user has required permissions
        return any(perm in user_permissions for perm in required_permissions)
    
    async def encrypt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive parameters"""
        if not self.encryption_enabled:
            return params
        
        # Identify sensitive fields
        sensitive_fields = ["file_path", "content", "text"]
        
        encrypted_params = params.copy()
        
        for field in sensitive_fields:
            if field in params:
                encrypted_params[field] = await self._encrypt_field(params[field])
        
        return encrypted_params
    
    async def _encrypt_field(self, value: Any) -> str:
        """Encrypt a single field value"""
        # Use Fernet symmetric encryption
        key = Fernet.generate_key()
        f = Fernet(key)
        
        # Convert to string if needed
        if not isinstance(value, str):
            value = json.dumps(value)
        
        encrypted_value = f.encrypt(value.encode())
        
        # Return base64 encoded encrypted value with key
        return base64.b64encode(key + encrypted_value).decode()
```

## Testing Strategies

### Unit Testing for MCP Components
```python
# tests/test_mcp_server.py

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.mcp_servers.classification_mcp_server import ClassificationMCPServer
from src.utils.mcp_contracts import ClassificationRequest

class TestClassificationMCPServer:
    
    @pytest.fixture
    async def server(self):
        """Create test server instance"""
        server = ClassificationMCPServer()
        await server.initialize()
        return server
    
    @pytest.fixture
    def sample_request(self):
        """Create sample classification request"""
        return {
            "request_id": "test-123",
            "file_path": "/tmp/test.pdf",
            "filename": "test.pdf",
            "classification_type": "automatic"
        }
    
    async def test_classify_document_success(self, server, sample_request):
        """Test successful document classification"""
        
        # Mock dependencies
        with patch.object(server.text_extractor, 'extract') as mock_extract, \
             patch.object(server.document_classifier, 'classify') as mock_classify, \
             patch.object(server.confidence_calculator, 'calculate') as mock_confidence:
            
            # Setup mocks
            mock_extract.return_value = "Sample document text"
            mock_classify.return_value = Mock(category="Invoice", method="AI", raw_output={})
            mock_confidence.return_value = 0.95
            
            # Execute
            result = await server.classify_document(sample_request)
            
            # Verify
            assert result["category"] == "Invoice"
            assert result["confidence"] == 0.95
            assert "processing_time" in result
            
            # Verify method calls
            mock_extract.assert_called_once()
            mock_classify.assert_called_once()
            mock_confidence.assert_called_once()
    
    async def test_classify_document_file_not_found(self, server, sample_request):
        """Test classification with missing file"""
        
        sample_request["file_path"] = "/nonexistent/file.pdf"
        
        with pytest.raises(MCPServerError) as exc_info:
            await server.classify_document(sample_request)
        
        assert "file not found" in str(exc_info.value).lower()
    
    async def test_classify_document_invalid_params(self, server):
        """Test classification with invalid parameters"""
        
        invalid_request = {"invalid": "params"}
        
        with pytest.raises(MCPServerError) as exc_info:
            await server.classify_document(invalid_request)
        
        assert "validation" in str(exc_info.value).lower()

# tests/test_mcp_orchestrator.py

class TestMCPOrchestrator:
    
    @pytest.fixture
    async def orchestrator(self):
        """Create test orchestrator instance"""
        orch = MCPOrchestrator()
        await orch.initialize()
        return orch
    
    @pytest.fixture
    def mock_server_config(self):
        """Create mock server configuration"""
        return ServerConfig(
            name="test_server",
            description="Test MCP Server",
            command=["python", "-m", "test_server"],
            capabilities=["test_capability"]
        )
    
    async def test_register_server(self, orchestrator, mock_server_config):
        """Test server registration"""
        
        await orchestrator.register_server(mock_server_config)
        
        assert "test_server" in orchestrator.servers
        assert "test_server" in orchestrator.connection_pools
        assert "test_server" in orchestrator.circuit_breakers
    
    async def test_send_request_success(self, orchestrator, mock_server_config):
        """Test successful request sending"""
        
        # Register mock server
        await orchestrator.register_server(mock_server_config)
        
        # Mock connection pool
        mock_client = Mock()
        mock_client.send_request.return_value = {"result": "success"}
        
        mock_pool = Mock()
        mock_pool.get_client.return_value = mock_client
        
        orchestrator.connection_pools["test_server"] = mock_pool
        
        # Execute request
        result = await orchestrator.send_request("test_server", "test_method", {})
        
        # Verify
        assert result == {"result": "success"}
        mock_client.send_request.assert_called_once_with("test_method", {})
    
    async def test_circuit_breaker_activation(self, orchestrator, mock_server_config):
        """Test circuit breaker activation on failures"""
        
        await orchestrator.register_server(mock_server_config)
        
        # Mock failing client
        mock_client = Mock()
        mock_client.send_request.side_effect = MCPServerError("Server error")
        
        mock_pool = Mock()
        mock_pool.get_client.return_value = mock_client
        
        orchestrator.connection_pools["test_server"] = mock_pool
        
        # Send multiple failing requests
        for _ in range(6):  # Exceed failure threshold
            with pytest.raises(MCPServerError):
                await orchestrator.send_request("test_server", "test_method", {})
        
        # Circuit breaker should be open now
        circuit_breaker = orchestrator.circuit_breakers["test_server"]
        assert circuit_breaker.is_open()
```

### Integration Testing
```python
# tests/integration/test_end_to_end.py

class TestEndToEndWorkflow:
    
    @pytest.fixture(scope="session")
    async def system_under_test(self):
        """Start complete system for integration testing"""
        
        # Start MCP servers
        classification_server = await start_classification_server()
        quality_server = await start_quality_server()
        
        # Start orchestrator
        orchestrator = MCPOrchestrator()
        await orchestrator.register_server(classification_server.config)
        await orchestrator.register_server(quality_server.config)
        
        # Start microservices
        classification_service = await start_classification_service(orchestrator)
        quality_service = await start_quality_service(orchestrator)
        
        # Start API gateway
        api_gateway = await start_api_gateway()
        
        yield {
            "orchestrator": orchestrator,
            "services": {
                "classification": classification_service,
                "quality": quality_service,
                "gateway": api_gateway
            }
        }
        
        # Cleanup
        await cleanup_all_services()
    
    async def test_document_classification_workflow(self, system_under_test):
        """Test complete document classification workflow"""
        
        # Prepare test document
        test_file = create_test_pdf("Sample invoice content")
        
        # Send request to API gateway
        async with aiohttp.ClientSession() as session:
            with aiofiles.open(test_file, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='test_invoice.pdf')
                
                async with session.post(
                    'http://localhost:8000/classify',
                    data=data
                ) as response:
                    
                    assert response.status == 200
                    result = await response.json()
                    
                    # Verify response structure
                    assert "request_id" in result
                    assert "filename" in result
                    assert "status" in result
                    assert "result" in result
                    
                    # Verify classification result
                    classification_result = result["result"]
                    assert "category" in classification_result
                    assert "confidence" in classification_result
                    assert classification_result["confidence"] > 0.0
    
    async def test_quality_analysis_workflow(self, system_under_test):
        """Test complete quality analysis workflow"""
        
        test_file = create_test_document("Poor quality document with issues")
        
        async with aiohttp.ClientSession() as session:
            with aiofiles.open(test_file, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='test_document.txt')
                
                async with session.post(
                    'http://localhost:8000/analyze',
                    data=data
                ) as response:
                    
                    assert response.status == 200
                    result = await response.json()
                    
                    # Verify quality analysis result
                    quality_result = result["result"]
                    assert "overall_score" in quality_result
                    assert "readability_score" in quality_result
                    assert "structure_score" in quality_result
                    assert "issues" in quality_result
                    assert "recommendations" in quality_result
```

---

**Document Version**: 3.0  
**Last Updated**: September 5, 2025  
**Technical Review**: Monthly  
**Implementation Status**: Active Production
