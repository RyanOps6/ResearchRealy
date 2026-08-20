import os
import tempfile
from src.rag.ast_parser import parse_python_file

def test_ast_python_parsing():
    """Verify that Python files are structurally parsed and chunked correctly by tree-sitter."""
    
    # 1. Define sample Python code content
    code = (
        "import os\n"
        "import sys\n"
        "\n"
        "GLOBAL_CONST = 42\n"
        "\n"
        "def standalone_add(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Calculator:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "\n"
        "    def add(self, x):\n"
        "        self.value += x\n"
        "        return self.value\n"
    )

    # 2. Write to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name

    try:
        # 3. Execute the parser
        references = parse_python_file(temp_path)
        
        # 4. Assertions on parsed contents
        assert len(references) > 0
        
        # Extract names for simpler search/assertions
        symbol_names = [ref.symbol_name for ref in references]
        
        # Verify standalone function
        assert "standalone_add" in symbol_names
        func_ref = next(ref for ref in references if ref.symbol_name == "standalone_add")
        assert func_ref.start_line == 6
        assert func_ref.end_line == 7
        assert "def standalone_add" in func_ref.code_snippet

        # Verify class block
        assert "Calculator" in symbol_names
        class_ref = next(ref for ref in references if ref.symbol_name == "Calculator")
        assert class_ref.start_line == 9
        assert class_ref.end_line == 15
        assert "class Calculator" in class_ref.code_snippet

        # Verify methods inside the class
        assert "Calculator.__init__" in symbol_names
        init_ref = next(ref for ref in references if ref.symbol_name == "Calculator.__init__")
        assert init_ref.start_line == 10
        assert init_ref.end_line == 11
        assert "self.value = 0" in init_ref.code_snippet

        assert "Calculator.add" in symbol_names
        add_ref = next(ref for ref in references if ref.symbol_name == "Calculator.add")
        assert add_ref.start_line == 13
        assert add_ref.end_line == 15
        assert "self.value += x" in add_ref.code_snippet

        # Verify global variables and imports
        assert "global" in symbol_names
        global_ref = next(ref for ref in references if ref.symbol_name == "global")
        assert global_ref.start_line == 1
        assert "GLOBAL_CONST = 42" in global_ref.code_snippet
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
