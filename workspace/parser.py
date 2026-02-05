"""
Parser for SimpleCalc Language
==============================

This module implements a parser that converts source code into an AST.

INPUT: Source code string (from .src files)
OUTPUT: AST as a Python dictionary (JSON-serializable)

The AST format is defined in task/grammar.ebnf
"""

from typing import Any

# ==============================================================================
# Token Class
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


# ==============================================================================
# Lexer (Tokenizer)
# ==============================================================================


class Lexer:
    """
    Tokenizer for SimpleCalc language.

    Token types:
        INTEGER, IDENTIFIER,
        PLUS, MINUS, STAR, SLASH,
        LPAREN, RPAREN, LBRACE, RBRACE,
        ASSIGN, SEMICOLON,
        EQ, NE, LT, GT, LE, GE,
        IF, ELSE, PRINT,
        EOF
    """

    KEYWORDS = {"if", "else", "print"}

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def _current_char(self) -> str | None:
        """Return current character or None if at end."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def _peek_char(self) -> str | None:
        """Peek at next character without consuming."""
        if self.pos + 1 >= len(self.source):
            return None
        return self.source[self.pos + 1]

    def _advance(self) -> str | None:
        """Consume and return current character."""
        ch = self._current_char()
        if ch is not None:
            self.pos += 1
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        return ch

    def _skip_whitespace(self):
        """Skip whitespace characters."""
        while self._current_char() is not None and self._current_char() in " \t\n\r":
            self._advance()

    def _skip_comment(self):
        """Skip // style comments."""
        if self._current_char() == "/" and self._peek_char() == "/":
            while self._current_char() is not None and self._current_char() != "\n":
                self._advance()
            if self._current_char() == "\n":
                self._advance()

    def _read_integer(self) -> Token:
        """Read an integer literal."""
        start_line, start_col = self.line, self.col
        result = ""
        while self._current_char() is not None and self._current_char().isdigit():
            result += self._advance()
        return Token("INTEGER", int(result), start_line, start_col)

    def _read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_line, start_col = self.line, self.col
        result = ""
        while self._current_char() is not None and (
            self._current_char().isalnum() or self._current_char() == "_"
        ):
            result += self._advance()

        # Check if it's a keyword
        if result in self.KEYWORDS:
            return Token(result.upper(), result, start_line, start_col)
        return Token("IDENTIFIER", result, start_line, start_col)

    def tokenize(self) -> list[Token]:
        """Convert source code into a list of tokens."""
        tokens = []

        while self._current_char() is not None:
            # Skip whitespace
            self._skip_whitespace()
            if self._current_char() is None:
                break

            # Skip comments
            if self._current_char() == "/" and self._peek_char() == "/":
                self._skip_comment()
                continue

            start_line, start_col = self.line, self.col
            ch = self._current_char()

            # Integer
            if ch.isdigit():
                tokens.append(self._read_integer())
            # Identifier or keyword
            elif ch.isalpha() or ch == "_":
                tokens.append(self._read_identifier())
            # Two-character operators
            elif ch == "=" and self._peek_char() == "=":
                self._advance()
                self._advance()
                tokens.append(Token("EQ", "==", start_line, start_col))
            elif ch == "!" and self._peek_char() == "=":
                self._advance()
                self._advance()
                tokens.append(Token("NE", "!=", start_line, start_col))
            elif ch == "<" and self._peek_char() == "=":
                self._advance()
                self._advance()
                tokens.append(Token("LE", "<=", start_line, start_col))
            elif ch == ">" and self._peek_char() == "=":
                self._advance()
                self._advance()
                tokens.append(Token("GE", ">=", start_line, start_col))
            # Single-character operators
            elif ch == "=":
                self._advance()
                tokens.append(Token("ASSIGN", "=", start_line, start_col))
            elif ch == "<":
                self._advance()
                tokens.append(Token("LT", "<", start_line, start_col))
            elif ch == ">":
                self._advance()
                tokens.append(Token("GT", ">", start_line, start_col))
            elif ch == "+":
                self._advance()
                tokens.append(Token("PLUS", "+", start_line, start_col))
            elif ch == "-":
                self._advance()
                tokens.append(Token("MINUS", "-", start_line, start_col))
            elif ch == "*":
                self._advance()
                tokens.append(Token("STAR", "*", start_line, start_col))
            elif ch == "/":
                self._advance()
                tokens.append(Token("SLASH", "/", start_line, start_col))
            elif ch == "(":
                self._advance()
                tokens.append(Token("LPAREN", "(", start_line, start_col))
            elif ch == ")":
                self._advance()
                tokens.append(Token("RPAREN", ")", start_line, start_col))
            elif ch == "{":
                self._advance()
                tokens.append(Token("LBRACE", "{", start_line, start_col))
            elif ch == "}":
                self._advance()
                tokens.append(Token("RBRACE", "}", start_line, start_col))
            elif ch == ";":
                self._advance()
                tokens.append(Token("SEMICOLON", ";", start_line, start_col))
            else:
                raise SyntaxError(
                    f"Unexpected character '{ch}' at line {self.line}, col {self.col}"
                )

        tokens.append(Token("EOF", None, self.line, self.col))
        return tokens


# ==============================================================================
# Parser (Recursive Descent)
# ==============================================================================


class Parser:
    """
    Recursive descent parser for SimpleCalc language.

    Grammar:
        program     = statement_list
        statement   = assignment_stmt | print_stmt | if_stmt
        assignment  = IDENTIFIER "=" expression ";"
        print_stmt  = "print" "(" expression ")" ";"
        if_stmt     = "if" "(" condition ")" "{" statement_list "}" ["else" "{" statement_list "}"]
        condition   = expression comparison_op expression
        expression  = term { ("+" | "-") term }
        term        = factor { ("*" | "/") factor }
        factor      = INTEGER | IDENTIFIER | "(" expression ")" | "-" factor
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        """Return current token."""
        return self.tokens[self.pos]

    def _peek(self) -> Token:
        """Peek at current token without consuming."""
        return self._current()

    def _advance(self) -> Token:
        """Consume and return current token."""
        token = self._current()
        if token.type != "EOF":
            self.pos += 1
        return token

    def _expect(self, token_type: str) -> Token:
        """Expect and consume a specific token type."""
        token = self._current()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type}, got {token.type} at line {token.line}, col {token.col}"
            )
        return self._advance()

    def _match(self, *types: str) -> bool:
        """Check if current token matches any of the given types."""
        return self._current().type in types

    def parse_program(self) -> dict:
        """Parse the entire program."""
        statements = self._parse_statement_list()
        return {"type": "Program", "statements": statements}

    def _parse_statement_list(self) -> list[dict]:
        """Parse a list of statements."""
        statements = []
        while not self._match("EOF", "RBRACE"):
            statements.append(self._parse_statement())
        return statements

    def _parse_statement(self) -> dict:
        """Parse a single statement."""
        if self._match("IF"):
            return self._parse_if_statement()
        elif self._match("PRINT"):
            return self._parse_print_statement()
        elif self._match("IDENTIFIER"):
            return self._parse_assignment_statement()
        else:
            raise SyntaxError(
                f"Unexpected token {self._current().type} at line {self._current().line}"
            )

    def _parse_assignment_statement(self) -> dict:
        """Parse: IDENTIFIER = expression ;"""
        var_token = self._expect("IDENTIFIER")
        self._expect("ASSIGN")
        value = self._parse_expression()
        self._expect("SEMICOLON")
        return {"type": "Assignment", "variable": var_token.value, "value": value}

    def _parse_print_statement(self) -> dict:
        """Parse: print ( expression ) ;"""
        self._expect("PRINT")
        self._expect("LPAREN")
        value = self._parse_expression()
        self._expect("RPAREN")
        self._expect("SEMICOLON")
        return {"type": "Print", "value": value}

    def _parse_if_statement(self) -> dict:
        """Parse: if ( condition ) { statements } [else { statements }]"""
        self._expect("IF")
        self._expect("LPAREN")
        condition = self._parse_condition()
        self._expect("RPAREN")
        self._expect("LBRACE")
        then_body = self._parse_statement_list()
        self._expect("RBRACE")

        else_body = []
        if self._match("ELSE"):
            self._advance()
            self._expect("LBRACE")
            else_body = self._parse_statement_list()
            self._expect("RBRACE")

        return {
            "type": "If",
            "condition": condition,
            "then_body": then_body,
            "else_body": else_body,
        }

    def _parse_condition(self) -> dict:
        """Parse: expression comparison_op expression"""
        left = self._parse_expression()

        if not self._match("EQ", "NE", "LT", "GT", "LE", "GE"):
            raise SyntaxError(f"Expected comparison operator, got {self._current().type}")

        op_token = self._advance()
        right = self._parse_expression()

        return {"type": "Condition", "operator": op_token.value, "left": left, "right": right}

    def _parse_expression(self) -> dict:
        """Parse: term { ("+" | "-") term }"""
        left = self._parse_term()

        while self._match("PLUS", "MINUS"):
            op_token = self._advance()
            right = self._parse_term()
            left = {"type": "BinaryOp", "operator": op_token.value, "left": left, "right": right}

        return left

    def _parse_term(self) -> dict:
        """Parse: factor { ("*" | "/") factor }"""
        left = self._parse_factor()

        while self._match("STAR", "SLASH"):
            op_token = self._advance()
            right = self._parse_factor()
            left = {"type": "BinaryOp", "operator": op_token.value, "left": left, "right": right}

        return left

    def _parse_factor(self) -> dict:
        """Parse: INTEGER | IDENTIFIER | "(" expression ")" | "-" factor"""
        if self._match("INTEGER"):
            token = self._advance()
            return {"type": "Integer", "value": token.value}

        elif self._match("IDENTIFIER"):
            token = self._advance()
            return {"type": "Variable", "name": token.value}

        elif self._match("LPAREN"):
            self._advance()
            expr = self._parse_expression()
            self._expect("RPAREN")
            return expr

        elif self._match("MINUS"):
            self._advance()
            operand = self._parse_factor()
            return {"type": "UnaryOp", "operator": "-", "operand": operand}

        else:
            raise SyntaxError(
                f"Unexpected token {self._current().type} in expression"
                f"at line {self._current().line}"
            )


# ==============================================================================
# Main Parse Function
# ==============================================================================


def parse(source_code: str) -> dict[str, Any]:
    """
    Parse source code into an AST.

    Args:
        source_code: The source code string to parse

    Returns:
        AST as a dictionary with the structure defined in grammar.ebnf

    Raises:
        SyntaxError: If the source code is invalid
    """
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse_program()


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
