"""
Example usage of MCP contracts
"""

import asyncio
from mcp.contracts import MCPRequest, MCPResponse, MCPContext

async def example_mcp_communication():
    # Create a request
    request = MCPRequest(
        method="classify_document",
        params={
            "file_path": "/path/to/document.pdf",
            "analysis_type": "full"
        }
    )
    
    # Convert to JSON (for transmission)
    request_json = request.to_json()
    print("Request JSON:", request_json)
    
    # Parse back from JSON (simulating receiving end)
    received_request = MCPRequest.from_json(request_json)
    print("Received request method:", received_request.method)
    
    # Create a response
    response = MCPResponse(
        id=request.id,  # Match the request ID
        result={
            "success": True,
            "classification": "invoice",
            "confidence": 0.95
        }
    )
    
    # Convert response to JSON
    response_json = response.to_json()
    print("Response JSON:", response_json)

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_mcp_communication())
