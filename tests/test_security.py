import os
import pytest
from src.core.security import validate_secure_path, scrub_secrets

def test_path_traversal_validation():
    """Verify that secure path validator stops traversal attempts outside project root."""
    # Setup temporary directory references
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(current_dir) # e:\TA

    # 1. Allowed Path: inside project root
    valid_path = os.path.join(project_root, "blueprints", "spec_1.md")
    resolved = validate_secure_path(valid_path, project_root)
    assert resolved == os.path.realpath(valid_path)

    # 2. Blocked Path: outside project root (traversal attempt)
    invalid_path = os.path.join(project_root, "..", "some_external_secret.txt")
    with pytest.raises(PermissionError) as exc_info:
        validate_secure_path(invalid_path, project_root)
    assert "outside root" in str(exc_info.value)

def test_secret_scrubbing():
    """Verify regex-based sanitization redacts credentials from logs and specifications."""
    
    # 1. Test OpenAI Key
    text_openai = "Connecting with API Key: sk-proj-1234567890123456789012345678901234567890"
    scrubbed_openai = scrub_secrets(text_openai)
    assert "sk-proj" not in scrubbed_openai
    assert "[REDACTED_OPENAI_KEY]" in scrubbed_openai

    # 2. Test Nvidia Key
    text_nvidia = "NIM key: nvapi-abcdefghijklmnopqrstu12345"
    scrubbed_nvidia = scrub_secrets(text_nvidia)
    assert "nvapi-" not in scrubbed_nvidia
    assert "[REDACTED_NVIDIA_KEY]" in scrubbed_nvidia

    # 3. Test Tavily Key
    text_tavily = "Search client key=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"
    scrubbed_tavily = scrub_secrets(text_tavily)
    assert "tvly-" not in scrubbed_tavily
    assert "[REDACTED_TAVILY_KEY]" in scrubbed_tavily

    # 4. Test Database connection string passwords
    text_postgres = "URI=postgresql://postgres:dbsecretpassword123@127.0.0.1:5432/agent_db"
    scrubbed_postgres = scrub_secrets(text_postgres)
    assert "dbsecretpassword123" not in scrubbed_postgres
    assert "postgres:[REDACTED_PASSWORD]@" in scrubbed_postgres

    # 5. Test Private RSA keys block
    private_key_block = (
        "Some header text\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Yxyz123abc456def789...\n"
        "-----END RSA PRIVATE KEY-----\n"
        "Some footer text"
    )
    scrubbed_key = scrub_secrets(private_key_block)
    assert "MIIEow" not in scrubbed_key
    assert "[REDACTED_PRIVATE_KEY]" in scrubbed_key
