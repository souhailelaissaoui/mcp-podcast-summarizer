# server.py basé sur examples https://github.com/modelcontextprotocol/python-sdk/tree/2ca2de767b316832fdcb96984dd53c5c4c80b3be/examples/fastmcp
from mcp.server.fastmcp import FastMCP
from transcription import download_audio, transcribe_audio_file
from security import require_api_key
import logging
import traceback
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create an MCP server
mcp = FastMCP("Podcast Analyzer")



# Simple error handling decorator
def handle_errors(func):
    """Decorator to handle errors and format them for LLM interpretation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logging.info(f"Starting {func.__name__} with args: {args[1:] if args else ''}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            
            # Check if result is already an error dict
            if isinstance(result, dict) and not result.get("success", True):
                error_msg = result.get("error", "Unknown error")
                logging.error(f"Error in {func.__name__}: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "tool_error",
                    "human_readable": f"The tool '{func.__name__}' encountered an error: {error_msg}. Please try again or use a different approach."
                }
            
            logging.info(f"Successfully completed {func.__name__}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logging.error(f"Exception in {func.__name__}: {error_msg}")
            logging.error(stack_trace)
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": "exception",
                "human_readable": f"An exception occurred while running '{func.__name__}': {error_msg}. This is likely a technical issue that needs to be fixed."
            }
    return wrapper

@mcp.tool()
@handle_errors
@require_api_key
def transcribe_audio(url):
    """
    Télécharge l'audio d'une vidéo YouTube et le transcrit.
    
    Args:
        url (str): URL de la vidéo YouTube
        
    Returns:
        dict: Dictionary containing transcription or error information
    """
    logging.info(f"Starting transcription for URL: {url}")
    
    # Téléchargement de l'audio
    audio_file, error = download_audio(url)
    if error:
        logging.error(f"Error downloading audio: {error}")
        return {"success": False, "error": error}
    
    logging.info(f"Audio downloaded successfully to {audio_file}")
    
    # Transcription
    transcript, error = transcribe_audio_file(audio_file)
    if error:
        logging.error(f"Error transcribing audio: {error}")
        return {"success": False, "error": error}
    
    logging.info(f"Transcription completed successfully")
    return {"success": True, "transcript": transcript}