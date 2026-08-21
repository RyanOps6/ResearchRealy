import os
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

def get_tracing_callbacks() -> List[Any]:
    """
    Dynamically loads and returns Langfuse tracing callback handlers
    if valid credentials exist in the environment.
    """
    callbacks = []
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if public_key and secret_key and public_key != "your-public-key" and secret_key != "your-secret-key":
        try:
            from langfuse.callback import CallbackHandler
            handler = CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            callbacks.append(handler)
            logger.info("Langfuse callback tracing handler initialized successfully.")
        except ImportError:
            logger.warning(
                "Langfuse keys are defined in the environment, but the 'langfuse' package "
                "is not installed. Skipping tracing callback initialization."
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse callback tracing: {e}")

    return callbacks
