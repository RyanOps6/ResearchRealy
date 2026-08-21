import os
import re

# High-fidelity regex patterns for credential detection
SECRET_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9-]{32,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"nvapi-[a-zA-Z0-9-]{20,}"), "[REDACTED_NVIDIA_KEY]"),
    (re.compile(r"tvly-[a-zA-Z0-9-]{20,}"), "[REDACTED_TAVILY_KEY]"),
    (re.compile(r"postgresql://(?P<user>[^:]+):(?P<pass>[^@]+)@"), r"postgresql://\g<user>:[REDACTED_PASSWORD]@"),
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[^-]+-----END [A-Z ]+ PRIVATE KEY-----", re.DOTALL), "[REDACTED_PRIVATE_KEY]"),
]

def validate_secure_path(path: str, project_root: str) -> str:
    """
    Resolves the real paths and validates that the target path is strictly
    located within the project root folder to prevent path traversal attacks.
    """
    abs_root = os.path.realpath(project_root)
    abs_path = os.path.realpath(path)

    # Standardize directory check with trailing separator
    root_prefix = abs_root if abs_root.endswith(os.sep) else abs_root + os.sep
    
    if not abs_path.startswith(root_prefix) and abs_path != abs_root:
        raise PermissionError(
            f"Security Exception: Path traversal attempt blocked. "
            f"Target path '{path}' resolves to '{abs_path}', which lies outside root '{abs_root}'"
        )
    return abs_path

def scrub_secrets(text: str) -> str:
    """
    Scrubs API keys, passwords, database credentials, and private keys
    from text logs and generated specifications.
    """
    if not text:
        return text

    scrubbed = text
    for pattern, replacement in SECRET_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed
