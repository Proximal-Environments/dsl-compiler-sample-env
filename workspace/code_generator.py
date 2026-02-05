"""
Code Generator for SimpleCalc Language
=======================================

This module implements a code generator that converts an AST into
assembly code for the SimpleVM architecture.

INPUT: AST dictionary (from parser.py)
OUTPUT: Assembly code string (for SimpleVM)

The instruction set is defined in task/instruction-set.txt
"""

from typing import Any

# ==============================================================================
# Symbol Table
# ==============================================================================


class SymbolTable:
    """
    Maps variable names to memory addresses.

    Memory layout (from instruction-set.txt):
        [0-63]    : Variable storage
        [64-127]  : Temporary values
        [128-255] : Reserved
    """

    def __init__(self):
        self.variables: dict[str, int] = {}
        self.next_addr = 0

    def get_address(self, name: str) -> int:
        """Get or allocate memory address for a variable."""
        if name not in self.variables:
            self.variables[name] = self.next_addr
            self.next_addr += 1
        return self.variables[name]


# ==============================================================================
# Code Generator
# ==============================================================================


class CodeGenerator:
    """
    Generates SimpleVM assembly from AST.

    Uses a stack-based approach for expression evaluation to handle
    arbitrarily complex expressions with only 4 registers.
    """

    def __init__(self):
        self.symbols = SymbolTable()
        self.code: list[str] = []
        self.label_counter = 0

    def new_label(self, prefix: str = "L") -> str:
        """Generate a unique label."""
        label = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return label

    def emit(self, instruction: str):
        """Emit an instruction."""
        self.code.append(instruction)

    def emit_label(self, label: str):
        """Emit a label definition."""
        self.code.append(f"{label}:")

    def generate_program(self, ast: dict) -> str:
        """Generate code for the entire program."""
        assert ast["type"] == "Program"

        for stmt in ast["statements"]:
            self.generate_statement(stmt)

        self.emit("HALT")
        return "\n".join(self.code)

    def generate_statement(self, stmt: dict):
        """Generate code for a statement."""
        stmt_type = stmt["type"]

        if stmt_type == "Assignment":
            self._generate_assignment(stmt)
        elif stmt_type == "Print":
            self._generate_print(stmt)
        elif stmt_type == "If":
            self._generate_if(stmt)
        else:
            raise ValueError(f"Unknown statement type: {stmt_type}")

    def _generate_assignment(self, stmt: dict):
        """Generate code for: variable = expression"""
        var_name = stmt["variable"]
        addr = self.symbols.get_address(var_name)

        # Evaluate expression into R0
        self.generate_expression(stmt["value"], "R0")

        # Store result to memory
        self.emit(f"STORE R0, [{addr}]")

    def _generate_print(self, stmt: dict):
        """Generate code for: print(expression)"""
        # Evaluate expression into R0
        self.generate_expression(stmt["value"], "R0")

        # Print the result
        self.emit("PRINT R0")

    def _generate_if(self, stmt: dict):
        """Generate code for: if (condition) { ... } [else { ... }]"""
        condition = stmt["condition"]
        then_body = stmt["then_body"]
        else_body = stmt["else_body"]

        # Generate labels
        else_label = self.new_label("else")
        end_label = self.new_label("endif")

        # Evaluate condition
        self._generate_condition(condition, else_label)

        # Generate then body
        for s in then_body:
            self.generate_statement(s)

        if else_body:
            self.emit(f"JMP {end_label}")

        # Generate else body
        self.emit_label(else_label)
        if else_body:
            for s in else_body:
                self.generate_statement(s)
            self.emit_label(end_label)

    def _generate_condition(self, cond: dict, false_label: str):
        """Generate code for a condition, jumping to false_label if false."""
        assert cond["type"] == "Condition"

        # Evaluate left operand into R0
        self.generate_expression(cond["left"], "R0")
        # Save R0 to stack
        self.emit("PUSH R0")

        # Evaluate right operand into R1
        self.generate_expression(cond["right"], "R1")

        # Restore left operand from stack
        self.emit("POP R0")

        # Compare
        self.emit("CMP R0, R1")

        # Jump based on operator (jump if condition is FALSE)
        op = cond["operator"]
        if op == "==":
            self.emit(f"JNE {false_label}")  # Jump if NOT equal
        elif op == "!=":
            self.emit(f"JEQ {false_label}")  # Jump if equal
        elif op == "<":
            self.emit(f"JGE {false_label}")  # Jump if >= (not less)
        elif op == ">":
            self.emit(f"JLE {false_label}")  # Jump if <= (not greater)
        elif op == "<=":
            self.emit(f"JGT {false_label}")  # Jump if > (not less or equal)
        elif op == ">=":
            self.emit(f"JLT {false_label}")  # Jump if < (not greater or equal)
        else:
            raise ValueError(f"Unknown comparison operator: {op}")

    def generate_expression(self, expr: dict, target_reg: str):
        """Generate code for an expression, result in target_reg."""
        expr_type = expr["type"]

        if expr_type == "Integer":
            self.emit(f"LOAD {target_reg}, #{expr['value']}")

        elif expr_type == "Variable":
            addr = self.symbols.get_address(expr["name"])
            self.emit(f"LOAD {target_reg}, [{addr}]")

        elif expr_type == "UnaryOp":
            self._generate_unary_op(expr, target_reg)

        elif expr_type == "BinaryOp":
            self._generate_binary_op(expr, target_reg)

        else:
            raise ValueError(f"Unknown expression type: {expr_type}")

    def _generate_unary_op(self, expr: dict, target_reg: str):
        """Generate code for unary operations."""
        assert expr["operator"] == "-"  # Only unary minus supported

        # Evaluate operand
        self.generate_expression(expr["operand"], target_reg)

        # Negate
        self.emit(f"NEG {target_reg}, {target_reg}")

    def _generate_binary_op(self, expr: dict, target_reg: str):
        """Generate code for binary operations."""
        op = expr["operator"]
        left = expr["left"]
        right = expr["right"]

        # Evaluate left operand into target register
        self.generate_expression(left, target_reg)

        # Save left result to stack
        self.emit(f"PUSH {target_reg}")

        # Evaluate right operand into R1 (or R0 if target is R1)
        temp_reg = "R1" if target_reg != "R1" else "R0"
        self.generate_expression(right, temp_reg)

        # Restore left operand from stack into R2
        self.emit("POP R2")

        # Perform operation: target = left op right
        op_instr = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[op]
        self.emit(f"{op_instr} {target_reg}, R2, {temp_reg}")


# ==============================================================================
# Main Generate Function
# ==============================================================================


def generate(ast: dict[str, Any]) -> str:
    """
    Generate assembly code from an AST.

    Args:
        ast: The AST dictionary from the parser

    Returns:
        Assembly code string for SimpleVM
    """
    generator = CodeGenerator()
    return generator.generate_program(ast)


# ==============================================================================
# For testing during development
# ==============================================================================

if __name__ == "__main__":
    # Test with a simple AST
    test_ast = {
        "type": "Program",
        "statements": [
            {
                "type": "Assignment",
                "variable": "x",
                "value": {
                    "type": "BinaryOp",
                    "operator": "+",
                    "left": {"type": "Integer", "value": 3},
                    "right": {"type": "Integer", "value": 5},
                },
            },
            {"type": "Print", "value": {"type": "Variable", "name": "x"}},
        ],
    }
    try:
        asm = generate(test_ast)
        print("Generated assembly:")
        print(asm)
    except NotImplementedError as e:
        print(f"Not implemented: {e}")
