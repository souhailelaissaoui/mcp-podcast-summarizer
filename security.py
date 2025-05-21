"""
Security module for MCP server authentication and authorization.

This module provides decorators and utilities for securing MCP tools
with API key authentication.
"""

import os
import logging
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug flag - can be set from main.py
DEBUG = False

def set_debug(enabled=False):
    """Enable or disable debug logging in the security module.
    
    Args:
        enabled (bool): Whether to enable debug logging
    """
    global DEBUG
    DEBUG = enabled
    logging.info(f"Security debug logging {'enabled' if enabled else 'disabled'}")

# Get server API key from environment
SERVER_API_KEY = os.getenv('SERVER_MCP_API_KEY')
if not SERVER_API_KEY:
    logging.warning("No SERVER_MCP_API_KEY found in environment variables. Server authentication will be disabled.")

def require_api_key(func):
    """Decorator to check for a valid API key before executing a tool.
    
    Args:
        func: The function to decorate
        
    Returns:
        The wrapped function that performs authentication before execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If SERVER_API_KEY is not set, skip authentication
        if not SERVER_API_KEY:
            if DEBUG:
                logging.info("API key authentication is disabled")
            return func(*args, **kwargs)
        
        # Get the API key from the environment passed by the client
        client_api_key = os.environ.get('CLIENT_MCP_API_KEY', '')
        
        if DEBUG:
            logging.info(f"Server API key: {SERVER_API_KEY}")
            logging.info(f"Client API key: {client_api_key}")
        
        # Compare the client API key with the server API key
        if client_api_key and client_api_key == SERVER_API_KEY:
            # API key is valid, proceed with the function
            if DEBUG:
                logging.info("API key authentication successful")
            return func(*args, **kwargs)
        
        # If we reach here, authentication failed
        logging.warning(f"Unauthorized access attempt to {func.__name__}")
        return {
            "success": False,
            "error": "Unauthorized access. Invalid or missing API key.",
            "error_type": "authentication_error",
            "human_readable": "Authentication failed. Please provide a valid SERVER_MCP_API_KEY in the environment."
        }
    
    return wrapper
