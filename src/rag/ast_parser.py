import logging
from typing import List, Optional
from tree_sitter import Parser
import tree_sitter_languages
from src.core.state import CodeReference

logger = logging.getLogger(__name__)

# Initialize tree-sitter Python parser
try:
    PY_LANGUAGE = tree_sitter_languages.get_language("python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
except Exception as e:
    logger.error(f"Failed to initialize tree-sitter language parser: {e}")
    parser = None

def parse_python_file(file_path: str) -> List[CodeReference]:
    """
    Parses a Python file using tree-sitter and returns a list of CodeReferences.
    Extracts structural chunks for classes, methods, functions, and global code.
    """
    if parser is None:
        logger.error("Tree-sitter parser is not initialized.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return []

    bytes_content = content.encode("utf-8")
    tree = parser.parse(bytes_content)
    
    classes = []
    functions = []  # list of tuples: (node, parent_class_name)

    def traverse(node, parent_class: Optional[str] = None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = name_node.text.decode("utf-8") if name_node else "UnknownClass"
            classes.append(node)
            for child in node.children:
                traverse(child, class_name)
        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            func_name = name_node.text.decode("utf-8") if name_node else "unknown_function"
            functions.append((node, parent_class))
            # Nested function definitions are ignored to avoid duplicate indexing
        else:
            for child in node.children:
                traverse(child, parent_class)

    traverse(tree.root_node)
    
    references = []
    covered_lines = set()
    
    # 1. Process Class declarations
    for node in classes:
        name_node = node.child_by_field_name("name")
        symbol = name_node.text.decode("utf-8") if name_node else "UnknownClass"
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        snippet = bytes_content[node.start_byte:node.end_byte].decode("utf-8")
        
        references.append(CodeReference(
            file_path=file_path,
            symbol_name=symbol,
            start_line=start_line,
            end_line=end_line,
            code_snippet=snippet
        ))
        for line in range(start_line, end_line + 1):
            covered_lines.add(line)
            
    # 2. Process Function/Method declarations
    for node, parent_class in functions:
        name_node = node.child_by_field_name("name")
        func_name = name_node.text.decode("utf-8") if name_node else "unknown_function"
        symbol = f"{parent_class}.{func_name}" if parent_class else func_name
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        snippet = bytes_content[node.start_byte:node.end_byte].decode("utf-8")
        
        references.append(CodeReference(
            file_path=file_path,
            symbol_name=symbol,
            start_line=start_line,
            end_line=end_line,
            code_snippet=snippet
        ))
        for line in range(start_line, end_line + 1):
            covered_lines.add(line)

    # 3. Process remaining uncovered global code blocks (imports, globals, inline script runner)
    lines = content.splitlines()
    total_lines = len(lines)
    
    uncovered_groups = []
    current_group = []
    
    for i in range(1, total_lines + 1):
        if i not in covered_lines:
            if lines[i - 1].strip():  # Skip empty lines
                current_group.append(i)
        else:
            if current_group:
                uncovered_groups.append(current_group)
                current_group = []
    if current_group:
        uncovered_groups.append(current_group)
        
    for group in uncovered_groups:
        # Slice continuous blocks into chunks of maximum 50 lines
        for offset in range(0, len(group), 50):
            sub_group = group[offset:offset + 50]
            start_line = sub_group[0]
            end_line = sub_group[-1]
            snippet = "\n".join(lines[start_line - 1:end_line])
            
            references.append(CodeReference(
                file_path=file_path,
                symbol_name="global",
                start_line=start_line,
                end_line=end_line,
                code_snippet=snippet
            ))
            
    return references
