# modifié depuis: https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py
import asyncio
import json
import logging
import os
import shutil
import argparse
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from security import set_debug

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.mcp_api_key = os.getenv("CLIENT_MCP_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
        timeout: float = 120.0,  # Increased timeout to 120 seconds for transcription tasks
    ) -> Any:
        """Execute a tool with retry mechanism and timeout.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.
            timeout: Timeout in seconds for the tool execution.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            TimeoutError: If tool execution times out.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name} with arguments: {arguments}")
                
                # Create a task for the tool execution
                tool_task = asyncio.create_task(self.session.call_tool(tool_name, arguments))
                
                # Wait for the task to complete with a timeout
                try:
                    # Log that we're waiting for the tool to complete
                    logging.info(f"Waiting for {tool_name} to complete (timeout: {timeout}s)...")
                    
                    # Wait for the task with timeout
                    result = await asyncio.wait_for(tool_task, timeout=timeout)
                    
                    # Log the result summary
                    if isinstance(result, dict):
                        if "success" in result:
                            if result["success"]:
                                logging.info(f"Tool {tool_name} executed successfully")
                            else:
                                logging.error(f"Tool {tool_name} failed: {result.get('error', 'Unknown error')}")
                        elif "error" in result:
                            logging.error(f"Tool {tool_name} returned error: {result['error']}")
                    else:
                        logging.info(f"Tool {tool_name} executed successfully with non-dict result")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    # Cancel the task if it times out
                    tool_task.cancel()
                    error_msg = f"Tool execution timed out after {timeout} seconds"
                    logging.error(error_msg)
                    
                    # Try to clean up any resources that might be hanging
                    try:
                        await asyncio.shield(self.session.cancel_tool())
                        logging.info("Sent tool cancellation request")
                    except Exception as cancel_error:
                        logging.warning(f"Error during tool cancellation: {cancel_error}")
                    
                    return {"success": False, "error": error_msg}

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    error_msg = f"Tool execution failed after {retries} attempts: {str(e)}"
                    logging.error(error_msg)
                    return {"success": False, "error": error_msg}

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            if self.session is None:
                logging.info(f"Server {self.name} already cleaned up")
                return
                
            try:
                logging.info(f"Closing session for server {self.name}...")
                
                # Close the exit stack with a timeout
                try:
                    # Create a task for closing the exit stack
                    close_task = asyncio.create_task(self.exit_stack.aclose())
                    
                    # Wait for the task to complete with a timeout
                    await asyncio.wait_for(close_task, timeout=5.0)
                    logging.info(f"Exit stack closed successfully for server {self.name}")
                except asyncio.TimeoutError:
                    logging.warning(f"Timeout while closing exit stack for server {self.name}")
                except Exception as close_error:
                    logging.warning(f"Error closing exit stack for server {self.name}: {close_error}")
                
                # Clear references
                self.session = None
                self.stdio_context = None
                logging.info(f"Server {self.name} cleaned up successfully")
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")
                # Still clear references even if cleanup failed
                self.session = None
                self.stdio_context = None


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from Claude.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Extract system message if present
        system_prompt = None
        api_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # For Claude API, system messages are not included in the messages array
                # but passed as a separate 'system' parameter
                system_prompt = msg["content"]
            else:
                # Format each message for Claude API
                # Claude expects 'user' and 'assistant' roles
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "temperature": 0.7,
            "max_tokens": 1024,  # Reduced from 4096 to improve response time
            "messages": api_messages,
            "stream": False
        }
        
        # Add system prompt if it exists
        if system_prompt:
            payload["system"] = system_prompt
            
        logging.debug(f"Sending payload to Claude API: {payload}")

        try:
            # Set a timeout for the request to prevent hanging
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Claude API returns content as a list of blocks
                content_blocks = data.get("content", [])
                # Extract text from the content blocks
                text_content = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                
                return text_content

        except httpx.TimeoutException:
            error_message = "Request to Claude API timed out after 30 seconds"
            logging.error(error_message)
            return f"I encountered a timeout error: {error_message}. Please try again later."
            
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        if not self.servers:
            return
            
        logging.info("Starting cleanup of servers...")
        for server in self.servers:
            try:
                logging.info(f"Cleaning up server {server.name}...")
                await server.cleanup()
                logging.info(f"Server {server.name} cleaned up successfully")
            except Exception as e:
                logging.warning(f"Warning during cleanup of server {server.name}: {e}")
        
        logging.info("All servers cleaned up")
        # Clear the servers list to prevent double cleanup
        self.servers = []

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            # Try to parse the response as JSON
            logging.info(f"Processing LLM response: {llm_response[:100]}...")
            
            # Check if the response looks like JSON
            if not (llm_response.strip().startswith('{') and llm_response.strip().endswith('}')):
                logging.info("Response is not JSON, returning as is")
                return llm_response
                
            tool_call = json.loads(llm_response)
            
            # Check if this is a tool call
            if "tool" in tool_call and "arguments" in tool_call:
                tool_name = tool_call['tool']
                arguments = tool_call['arguments']
                
                logging.info(f"Detected tool call: {tool_name}")
                logging.info(f"With arguments: {arguments}")
                
                # Print a clear separator for better log readability
                logging.info("="*50)
                logging.info(f"EXECUTING TOOL: {tool_name}")
                logging.info("="*50)

                # Find the server that has this tool
                tool_server = None
                for server in self.servers:
                    try:
                        tools = await server.list_tools()
                        if any(tool.name == tool_name for tool in tools):
                            tool_server = server
                            break
                    except Exception as e:
                        logging.error(f"Error listing tools from server {server.name}: {e}")
                
                if tool_server:
                    try:
                        logging.info(f"Found tool {tool_name} on server {tool_server.name}, executing...")
                        
                        # Execute the tool
                        result = await tool_server.execute_tool(tool_name, arguments)
                        
                        # Handle progress reporting
                        if isinstance(result, dict):
                            if "progress" in result and "total" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(f"Progress: {progress}/{total} ({percentage:.1f}%)")
                            
                            # Format the result for better readability
                            if "success" in result:
                                if result["success"]:
                                    if "transcript" in result:
                                        # For transcription results, truncate if too long
                                        transcript = result["transcript"]
                                        transcript_preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
                                        logging.info(f"Transcription successful, length: {len(transcript)} chars")
                                        return f"Tool execution successful. Transcription result: {transcript_preview}"
                                    else:
                                        return f"Tool execution successful. Result: {result}"
                                else:
                                    # Handle the enhanced error format
                                    error = result.get("error", "Unknown error")
                                    error_type = result.get("error_type", "tool_error")
                                    human_readable = result.get("human_readable", f"Tool execution failed: {error}")
                                    
                                    logging.error(f"Tool execution failed: {error} (Type: {error_type})")
                                    
                                    # Return the human-readable message if available
                                    return human_readable
                        
                        # Default formatting for other result types
                        logging.info(f"Tool execution completed with result: {result}")
                        return f"Tool execution result: {result}"
                        
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        logging.error(error_msg)
                        return error_msg
                else:
                    error_msg = f"No server found with tool: {tool_name}"
                    logging.error(error_msg)
                    return error_msg
            else:
                # Not a tool call, return the response as is
                return llm_response
        except json.JSONDecodeError as e:
            logging.info(f"Response is not valid JSON: {e}")
            return llm_response
        except Exception as e:
            logging.error(f"Unexpected error processing LLM response: {e}")
            return f"Error processing response: {str(e)}. Original response: {llm_response}"

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            # Initialize servers
            logging.info("Initializing servers...")
            for server in self.servers:
                try:
                    logging.info(f"Initializing server {server.name}...")
                    await server.initialize()
                    logging.info(f"Server {server.name} initialized successfully")
                except Exception as e:
                    logging.error(f"Failed to initialize server {server.name}: {e}")
                    await self.cleanup_servers()
                    return

            # List available tools
            logging.info("Listing available tools...")
            all_tools = []
            for server in self.servers:
                try:
                    tools = await server.list_tools()
                    logging.info(f"Found {len(tools)} tools on server {server.name}")
                    all_tools.extend(tools)
                except Exception as e:
                    logging.error(f"Failed to list tools from server {server.name}: {e}")
                    # Continue with other servers rather than failing completely

            if not all_tools:
                logging.error("No tools found on any server. Exiting.")
                await self.cleanup_servers()
                return

            # Format tools description for system message
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            logging.info(f"Prepared descriptions for {len(all_tools)} tools")

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = [{"role": "system", "content": system_message}]
            logging.info("Chat session initialized successfully")
            logging.info("Starting chat loop. Type 'quit' or 'exit' to end the session.")

            while True:
                try:
                    print("\nYou: ", end="", flush=True)
                    user_input = input().strip()
                    
                    if user_input.lower() in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})
                    logging.info(f"Processing user input: {user_input[:50]}..." if len(user_input) > 50 else f"Processing user input: {user_input}")

                    # Get LLM response
                    try:
                        logging.info("Getting response from LLM...")
                        llm_response = self.llm_client.get_response(messages)
                        logging.info(f"Received LLM response: {llm_response[:100]}..." if len(llm_response) > 100 else f"Received LLM response: {llm_response}")
                        print(f"\nAssistant: {llm_response}")
                    except Exception as e:
                        error_msg = f"Error getting LLM response: {str(e)}"
                        logging.error(error_msg)
                        print(f"\nAssistant: {error_msg}")
                        continue

                    # Process tool calls if present
                    try:
                        result = await self.process_llm_response(llm_response)
                        
                        if result != llm_response:
                            logging.info("Tool was executed, getting final response...")
                            messages.append({"role": "assistant", "content": llm_response})
                            messages.append({"role": "system", "content": result})

                            try:
                                final_response = self.llm_client.get_response(messages)
                                logging.info(f"Final response: {final_response[:100]}..." if len(final_response) > 100 else f"Final response: {final_response}")
                                print(f"\nAssistant: {final_response}")
                                messages.append({"role": "assistant", "content": final_response})
                            except Exception as e:
                                error_msg = f"Error getting final response: {str(e)}"
                                logging.error(error_msg)
                                print(f"\nAssistant: {error_msg}")
                                messages.append({"role": "assistant", "content": result})
                        else:
                            messages.append({"role": "assistant", "content": llm_response})
                    except Exception as e:
                        error_msg = f"Error processing LLM response: {str(e)}"
                        logging.error(error_msg)
                        print(f"\nAssistant: {error_msg}")
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nKeyboard interrupt detected. Exiting...")
                    print("\nExiting due to keyboard interrupt...")
                    break
                except EOFError:
                    logging.info("\nEOF detected. Exiting...")
                    print("\nExiting due to EOF...")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error in chat loop: {str(e)}")
                    print(f"\nAn unexpected error occurred: {str(e)}")
                    # Continue the loop rather than breaking

        except Exception as e:
            logging.error(f"Critical error in chat session: {str(e)}")
            print(f"\nA critical error occurred: {str(e)}")
        finally:
            logging.info("Cleaning up resources...")
            await self.cleanup_servers()
            logging.info("Chat session ended")


async def main() -> None:
    """Initialize and run the chat session."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Simple Chatbot")
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging for security module"
    )
    args = parser.parse_args()
    
    # Configure debug mode if requested
    if args.debug:
        set_debug(enabled=True)
    
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
