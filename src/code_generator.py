"""
Code Generator for SimpleCalc Language
=======================================

This module should implement a code generator that converts an AST into
assembly code for the SimpleVM architecture.

INPUT: AST dictionary (from parser.py)
OUTPUT: Assembly code string (for SimpleVM)

The instruction set is defined in task/instruction-set.txt

IMPORTANT: The agent must implement the `generate` function below.
"""

from typing import Any

# ==============================================================================
# SimpleVM Instruction Reference (see task/instruction-set.txt for details)
# ==============================================================================
#
# Data Movement:
#   LOAD Rd, #imm       - Load immediate into register
#   LOAD Rd, [addr]     - Load from memory
#   STORE Rs, [addr]    - Store to memory
#   MOV Rd, Rs          - Copy register
#
# Arithmetic:
#   ADD Rd, Rs1, Rs2    - Rd = Rs1 + Rs2
#   SUB Rd, Rs1, Rs2    - Rd = Rs1 - Rs2
#   MUL Rd, Rs1, Rs2    - Rd = Rs1 * Rs2
#   DIV Rd, Rs1, Rs2    - Rd = Rs1 / Rs2
#   NEG Rd, Rs          - Rd = -Rs
#
# Comparison:
#   CMP Rs1, Rs2        - Compare, set flags
#
# Control Flow:
#   JMP label           - Unconditional jump
#   JEQ label           - Jump if equal
#   JNE label           - Jump if not equal
#   JLT label           - Jump if less than
#   JGT label           - Jump if greater than
#   JLE label           - Jump if less or equal
#   JGE label           - Jump if greater or equal
#
# Stack:
#   PUSH Rs             - Push register to stack
#   POP Rd              - Pop stack to register
#
# I/O:
#   PRINT Rs            - Output register value
#
# Control:
#   HALT                - Stop execution
#
# ==============================================================================


def generate(ast: dict[str, Any]) -> str:
    """
    Generate assembly code from an AST.

    Args:
        ast: The AST dictionary from the parser

    Returns:
        Assembly code string for SimpleVM

    Example:
        >>> ast = {
        ...     "type": "Program",
        ...     "statements": [
        ...         {
        ...             "type": "Assignment",
        ...             "variable": "x",
        ...             "value": {"type": "Integer", "value": 5}
        ...         }
        ...     ]
        ... }
        >>> print(generate(ast))
        LOAD R0, #5
        STORE R0, [0]
        HALT
    """
    # TODO: Implement this function
    #
    # Implementation hints:
    # 1. Traverse the AST recursively
    # 2. Maintain a symbol table mapping variable names to memory addresses
    # 3. Use registers R0-R3 for computation (need register allocation)
    # 4. Generate appropriate instructions for each AST node type
    # 5. Handle expression evaluation (may need stack for complex expressions)
    # 6. Generate labels for control flow (if-else)
    # 7. Always end with HALT
    #
    raise NotImplementedError("Code generator not yet implemented - this is your task!")


# ==============================================================================
# Optional: Helper classes for implementation
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


class RegisterAllocator:
    """
    Simple register allocator for R0-R3.

    Strategy: Use R0 for results, R1-R3 for temporaries.
    For complex expressions, use the stack.
    """

    def __init__(self):
        self.registers = ['R0', 'R1', 'R2', 'R3']
        self.in_use = [False, False, False, False]

    def allocate(self) -> str:
        """Allocate a free register."""
        for i, used in enumerate(self.in_use):
            if not used:
                self.in_use[i] = True
                return self.registers[i]
        raise RuntimeError("No free registers - use stack")

    def free(self, reg: str):
        """Free a register."""
        idx = self.registers.index(reg)
        self.in_use[idx] = False

    def free_all(self):
        """Free all registers."""
        self.in_use = [False, False, False, False]


class CodeGenerator:
    """
    Generates SimpleVM assembly from AST.
    """

    def __init__(self):
        self.symbols = SymbolTable()
        self.registers = RegisterAllocator()
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
        # TODO: Implement code generation
        raise NotImplementedError()

    def generate_statement(self, stmt: dict):
        """Generate code for a statement."""
        # TODO: Implement for each statement type
        raise NotImplementedError()

    def generate_expression(self, expr: dict, target_reg: str):
        """Generate code for an expression, result in target_reg."""
        # TODO: Implement for each expression type
        raise NotImplementedError()


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
                    "right": {"type": "Integer", "value": 5}
                }
            },
            {
                "type": "Print",
                "value": {"type": "Variable", "name": "x"}
            }
        ]
    }
    try:
        asm = generate(test_ast)
        print("Generated assembly:")
        print(asm)
    except NotImplementedError as e:
        print(f"Not implemented: {e}")
