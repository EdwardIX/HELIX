import ast

class DocstringRemover(ast.NodeTransformer):
    """Delete the first string literal docstring of modules, classes, and functions."""

    def _strip(self, body):
        # If the first statement of body is a string literal (docstring), delete it
        if body and isinstance(body[0], ast.Expr):
            v = body[0].value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return body[1:]
        return body

    def visit_Module(self, node: ast.Module):
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node


def normalize_python_code(code: str) -> str:
    """
    Standardize Python code:
    1. Delete all comments (AST output naturally has no comments)
    2. Delete all docstrings (modules, classes, functions)
    3. Use ast.unparse to output standard format
    """
    try:
        # Parse AST
        tree = ast.parse(code)

        # Delete docstring
        tree = DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)

        # Output formatted code
        normalized = ast.unparse(tree)

        # Ensure newline at end of file
        if not normalized.endswith("\n"):
            normalized += "\n"
        return normalized
    except Exception as e:
        print(f"Warning: Error in Normalizing Python Code: {type(e)}: {str(e)}")
        return code


# ---  ---
if __name__ == "__main__":
    messy_code = r'''
def normalize_python_code(code: str) -> str:
    """
    Standardize Python code:
    1. Delete all comments (AST output naturally has no comments)
    2. Delete all docstrings (modules, classes, functions)
    3. Use ast.unparse to output standard format
    """
    # Parse AST
    tree = ast.parse(code)

    # Delete docstring
    tree = DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)

    # Output formatted code
    normalized = ast.unparse(tree)

    # Ensure newline at end of file
    if not normalized.endswith('\n'):
        normalized += '\n'
    return normalized
# Top comment
def add( a,  b ):
    """sum two numbers"""
    # This is internal comment
    return   a +  b   # trailing
'''
    print(normalize_python_code(messy_code))
