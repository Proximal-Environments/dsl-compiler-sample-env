"""
Parser for SimpleCalc Language
==============================

This module should implement a parser that converts source code into an AST.

INPUT: Source code string (from .src files)
OUTPUT: AST as a Python dictionary (JSON-serializable)

The AST format is defined in task/grammar.ebnf

IMPORTANT: The agent must implement the `parse` function below.
"""

from typing import Any

# ==============================================================================
# AST Node Types (for reference)
# ==============================================================================
#
# Program:
#   { "type": "Program", "statements": [...] }
#
# AssignmentStatement:
#   { "type": "Assignment", "variable": "x", "value": <expression> }
#
# PrintStatement:
#   { "type": "Print", "value": <expression> }
#
# IfStatement:
#   { "type": "If", "condition": <condition>, "then_body": [...], "else_body": [...] }
#
# BinaryExpression:
#   { "type": "BinaryOp", "operator": "+", "left": <expr>, "right": <expr> }
#
# UnaryExpression:
#   { "type": "UnaryOp", "operator": "-", "operand": <expr> }
#
# IntegerLiteral:
#   { "type": "Integer", "value": 42 }
#
# Variable:
#   { "type": "Variable", "name": "x" }
#
# Condition:
#   { "type": "Condition", "operator": "<", "left": <expr>, "right": <expr> }
#
# ==============================================================================


def parse(source_code: str) -> dict[str, Any]:
    """
    Parse source code into an AST.

    Args:
        source_code: The source code string to parse

    Returns:
        AST as a dictionary with the structure defined above

    Raises:
        SyntaxError: If the source code is invalid

    Example:
        >>> parse("x = 5;")
        {
            "type": "Program",
            "statements": [
                {
                    "type": "Assignment",
                    "variable": "x",
                    "value": {"type": "Integer", "value": 5}
                }
            ]
        }
    """
    # TODO: Implement this function
    #
    # Implementation hints:
    # 1. Tokenize the source code (lexical analysis)
    # 2. Build a recursive descent parser based on the grammar
    # 3. Handle operator precedence correctly (* / before + -)
    # 4. Return the AST as a dictionary
    #
    raise NotImplementedError("Parser not yet implemented - this is your task!")


# ==============================================================================
# Optional: Helper classes for implementation
# ==============================================================================


class Token:
    """Represents a lexical token."""

    def __init__(self, type_: str, value: Any, line: int = 0, col: int = 0):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """
    Tokenizer for SimpleCalc language.

    Token types:
        INTEGER, IDENTIFIER,
        PLUS, MINUS, STAR, SLASH,
        LPAREN, RPAREN, LBRACE, RBRACE,
        ASSIGN, SEMICOLON, COMMA,
        EQ, NE, LT, GT, LE, GE,
        IF, ELSE, PRINT,
        EOF
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def tokenize(self) -> list[Token]:
        """Convert source code into a list of tokens."""
        # TODO: Implement tokenization
        raise NotImplementedError()


class Parser:
    """
    Recursive descent parser for SimpleCalc language.
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse_program(self) -> dict:
        """Parse the entire program."""
        # TODO: Implement parsing
        raise NotImplementedError()


# ==============================================================================
# For testing during development
# ==============================================================================

if __name__ == "__main__":
    test_code = """
    x = 3 + 5;
    print(x);
    """

    try:
        ast = parse(test_code)
        import json

        print(json.dumps(ast, indent=2))
    except NotImplementedError as e:
        print(f"Not implemented: {e}")
