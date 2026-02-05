"""
SimpleVM Emulator
=================

A fully functional emulator for the SimpleVM instruction set.
Executes assembly code and returns the final machine state.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmulatorState:
    """Represents the complete state of the emulator."""
    registers: dict[str, int] = field(default_factory=lambda: {"R0": 0, "R1": 0, "R2": 0, "R3": 0})
    memory: list[int] = field(default_factory=lambda: [0] * 256)
    stack: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)
    pc: int = 0  # Program counter
    halted: bool = False
    zero_flag: bool = False
    neg_flag: bool = False

    def copy(self) -> "EmulatorState":
        """Create a deep copy of the state."""
        return EmulatorState(
            registers=dict(self.registers),
            memory=list(self.memory),
            stack=list(self.stack),
            output=list(self.output),
            pc=self.pc,
            halted=self.halted,
            zero_flag=self.zero_flag,
            neg_flag=self.neg_flag,
        )


class EmulatorError(Exception):
    """Error during emulation."""
    pass


class SimpleVMEmulator:
    """
    Emulator for SimpleVM instruction set.

    Usage:
        emulator = SimpleVMEmulator()
        state = emulator.run(assembly_code)
        print(state.output)
    """

    MAX_INSTRUCTIONS = 100000  # Prevent infinite loops

    def __init__(self):
        self.state = EmulatorState()
        self.instructions: list[tuple[str, list[str]]] = []
        self.labels: dict[str, int] = {}

    def run(self, assembly: str) -> EmulatorState:
        """
        Execute assembly code and return final state.

        Args:
            assembly: Assembly code string

        Returns:
            Final EmulatorState after execution

        Raises:
            EmulatorError: If execution fails
        """
        self.state = EmulatorState()
        self._parse_assembly(assembly)
        self._execute()
        return self.state

    def _parse_assembly(self, assembly: str):
        """Parse assembly code into instructions and labels."""
        self.instructions = []
        self.labels = {}

        lines = assembly.strip().split('\n')
        instruction_index = 0

        for line in lines:
            # Remove comments
            if '//' in line:
                line = line[:line.index('//')]
            if ';' in line:
                line = line[:line.index(';')]
            line = line.strip()

            if not line:
                continue

            # Check for label
            if line.endswith(':'):
                label = line[:-1].strip()
                self.labels[label] = instruction_index
                continue

            # Parse instruction
            parts = self._tokenize_instruction(line)
            if parts:
                opcode = parts[0].upper()
                args = parts[1:] if len(parts) > 1 else []
                self.instructions.append((opcode, args))
                instruction_index += 1

    def _tokenize_instruction(self, line: str) -> list[str]:
        """Tokenize an instruction line."""
        # Replace commas with spaces for easier splitting
        line = line.replace(',', ' ')
        parts = line.split()
        return [p.strip() for p in parts if p.strip()]

    def _execute(self):
        """Execute all instructions."""
        instruction_count = 0

        while not self.state.halted and self.state.pc < len(self.instructions):
            if instruction_count >= self.MAX_INSTRUCTIONS:
                raise EmulatorError(f"Exceeded maximum instructions ({self.MAX_INSTRUCTIONS})")

            opcode, args = self.instructions[self.state.pc]
            self._execute_instruction(opcode, args)
            instruction_count += 1

        if not self.state.halted:
            raise EmulatorError("Program did not halt properly (missing HALT instruction)")

    def _execute_instruction(self, opcode: str, args: list[str]):
        """Execute a single instruction."""
        # Increment PC by default (jumps will override)
        next_pc = self.state.pc + 1

        try:
            if opcode == "LOAD":
                self._exec_load(args)
            elif opcode == "STORE":
                self._exec_store(args)
            elif opcode == "MOV":
                self._exec_mov(args)
            elif opcode == "ADD":
                self._exec_add(args)
            elif opcode == "SUB":
                self._exec_sub(args)
            elif opcode == "MUL":
                self._exec_mul(args)
            elif opcode == "DIV":
                self._exec_div(args)
            elif opcode == "NEG":
                self._exec_neg(args)
            elif opcode == "CMP":
                self._exec_cmp(args)
            elif opcode == "JMP":
                next_pc = self._exec_jmp(args)
            elif opcode == "JEQ":
                next_pc = self._exec_jeq(args)
            elif opcode == "JNE":
                next_pc = self._exec_jne(args)
            elif opcode == "JLT":
                next_pc = self._exec_jlt(args)
            elif opcode == "JGT":
                next_pc = self._exec_jgt(args)
            elif opcode == "JLE":
                next_pc = self._exec_jle(args)
            elif opcode == "JGE":
                next_pc = self._exec_jge(args)
            elif opcode == "PUSH":
                self._exec_push(args)
            elif opcode == "POP":
                self._exec_pop(args)
            elif opcode == "PRINT":
                self._exec_print(args)
            elif opcode == "HALT":
                self.state.halted = True
            elif opcode == "NOP":
                pass
            else:
                raise EmulatorError(f"Unknown opcode: {opcode}")
        except Exception as e:
            raise EmulatorError(f"Error executing {opcode} {args}: {e}")

        self.state.pc = next_pc

    def _parse_operand(self, operand: str) -> tuple[str, Any]:
        """
        Parse an operand and return (type, value).

        Types: 'reg', 'imm', 'mem', 'indirect', 'label'
        """
        operand = operand.strip()
        if operand.startswith('#'):
            # Immediate value
            return ('imm', int(operand[1:]))
        elif operand.startswith('[') and operand.endswith(']'):
            # Memory address
            return ('mem', int(operand[1:-1]))
        elif operand.startswith('@'):
            # Indirect addressing
            return ('indirect', operand[1:])
        elif operand.upper() in ('R0', 'R1', 'R2', 'R3'):
            # Register
            return ('reg', operand.upper())
        else:
            # Label
            return ('label', operand)

    def _get_value(self, operand: str) -> int:
        """Get the value of an operand."""
        op_type, value = self._parse_operand(operand)

        if op_type == 'imm':
            return value
        elif op_type == 'reg':
            return self.state.registers[value]
        elif op_type == 'mem':
            return self.state.memory[value]
        elif op_type == 'indirect':
            addr = self.state.registers[value]
            return self.state.memory[addr]
        else:
            raise EmulatorError(f"Cannot get value of: {operand}")

    def _exec_load(self, args: list[str]):
        """LOAD Rd, source"""
        rd = args[0].upper()
        source = args[1]
        self.state.registers[rd] = self._get_value(source)

    def _exec_store(self, args: list[str]):
        """STORE Rs, destination"""
        rs = args[0].upper()
        dest = args[1]
        value = self.state.registers[rs]

        op_type, addr = self._parse_operand(dest)
        if op_type == 'mem':
            self.state.memory[addr] = value
        elif op_type == 'indirect':
            mem_addr = self.state.registers[addr]
            self.state.memory[mem_addr] = value
        else:
            raise EmulatorError(f"Invalid STORE destination: {dest}")

    def _exec_mov(self, args: list[str]):
        """MOV Rd, Rs"""
        rd = args[0].upper()
        rs = args[1].upper()
        self.state.registers[rd] = self.state.registers[rs]

    def _exec_add(self, args: list[str]):
        """ADD Rd, Rs1, Rs2 or ADD Rd, Rs, #imm"""
        rd = args[0].upper()
        val1 = self._get_value(args[1])
        val2 = self._get_value(args[2])
        self.state.registers[rd] = val1 + val2

    def _exec_sub(self, args: list[str]):
        """SUB Rd, Rs1, Rs2 or SUB Rd, Rs, #imm"""
        rd = args[0].upper()
        val1 = self._get_value(args[1])
        val2 = self._get_value(args[2])
        self.state.registers[rd] = val1 - val2

    def _exec_mul(self, args: list[str]):
        """MUL Rd, Rs1, Rs2 or MUL Rd, Rs, #imm"""
        rd = args[0].upper()
        val1 = self._get_value(args[1])
        val2 = self._get_value(args[2])
        self.state.registers[rd] = val1 * val2

    def _exec_div(self, args: list[str]):
        """DIV Rd, Rs1, Rs2 or DIV Rd, Rs, #imm"""
        rd = args[0].upper()
        val1 = self._get_value(args[1])
        val2 = self._get_value(args[2])
        if val2 == 0:
            raise EmulatorError("Division by zero")
        # Integer division truncating toward zero
        self.state.registers[rd] = int(val1 / val2)

    def _exec_neg(self, args: list[str]):
        """NEG Rd, Rs"""
        rd = args[0].upper()
        rs = args[1].upper()
        self.state.registers[rd] = -self.state.registers[rs]

    def _exec_cmp(self, args: list[str]):
        """CMP Rs1, Rs2 or CMP Rs, #imm"""
        val1 = self._get_value(args[0])
        val2 = self._get_value(args[1])
        diff = val1 - val2
        self.state.zero_flag = (diff == 0)
        self.state.neg_flag = (diff < 0)

    def _get_label_pc(self, label: str) -> int:
        """Get PC for a label."""
        if label not in self.labels:
            raise EmulatorError(f"Unknown label: {label}")
        return self.labels[label]

    def _exec_jmp(self, args: list[str]) -> int:
        """JMP label"""
        return self._get_label_pc(args[0])

    def _exec_jeq(self, args: list[str]) -> int:
        """JEQ label - jump if equal"""
        if self.state.zero_flag:
            return self._get_label_pc(args[0])
        return self.state.pc + 1

    def _exec_jne(self, args: list[str]) -> int:
        """JNE label - jump if not equal"""
        if not self.state.zero_flag:
            return self._get_label_pc(args[0])
        return self.state.pc + 1

    def _exec_jlt(self, args: list[str]) -> int:
        """JLT label - jump if less than"""
        if self.state.neg_flag:
            return self._get_label_pc(args[0])
        return self.state.pc + 1

    def _exec_jgt(self, args: list[str]) -> int:
        """JGT label - jump if greater than"""
        if not self.state.zero_flag and not self.state.neg_flag:
            return self._get_label_pc(args[0])
        return self.state.pc + 1

    def _exec_jle(self, args: list[str]) -> int:
        """JLE label - jump if less than or equal"""
        if self.state.zero_flag or self.state.neg_flag:
            return self._get_label_pc(args[0])
        return self.state.pc + 1

    def _exec_jge(self, args: list[str]) -> int:
        """JGE label - jump if greater than or equal"""
        if self.state.zero_flag or not self.state.neg_flag:
            return self._get_label_pc(args[0])
        return self.state.pc + 1

    def _exec_push(self, args: list[str]):
        """PUSH Rs"""
        rs = args[0].upper()
        if len(self.state.stack) >= 64:
            raise EmulatorError("Stack overflow")
        self.state.stack.append(self.state.registers[rs])

    def _exec_pop(self, args: list[str]):
        """POP Rd"""
        rd = args[0].upper()
        if not self.state.stack:
            raise EmulatorError("Stack underflow")
        self.state.registers[rd] = self.state.stack.pop()

    def _exec_print(self, args: list[str]):
        """PRINT Rs"""
        rs = args[0].upper()
        self.state.output.append(self.state.registers[rs])


def run_assembly(assembly: str) -> EmulatorState:
    """
    Convenience function to run assembly code.

    Args:
        assembly: Assembly code string

    Returns:
        Final EmulatorState
    """
    emulator = SimpleVMEmulator()
    return emulator.run(assembly)


def extract_variables(state: EmulatorState, symbol_table: dict[str, int]) -> dict[str, int]:
    """
    Extract variable values from emulator state using a symbol table.

    Args:
        state: The emulator state
        symbol_table: Mapping of variable names to memory addresses

    Returns:
        Dictionary of variable names to their values
    """
    return {name: state.memory[addr] for name, addr in symbol_table.items()}


# ==============================================================================
# For standalone testing
# ==============================================================================

if __name__ == "__main__":
    # Test the emulator with a simple program
    test_asm = """
        ; Calculate (3 + 5) * 2 and print
        LOAD R0, #3       ; R0 = 3
        LOAD R1, #5       ; R1 = 5
        ADD R2, R0, R1    ; R2 = 8
        LOAD R3, #2       ; R3 = 2
        MUL R0, R2, R3    ; R0 = 16
        STORE R0, [0]     ; memory[0] = 16
        PRINT R0          ; output: 16
        HALT
    """

    print("Testing emulator with: (3 + 5) * 2")
    print("=" * 50)

    try:
        state = run_assembly(test_asm)
        print(f"Registers: {state.registers}")
        print(f"Memory[0-3]: {state.memory[0:4]}")
        print(f"Output: {state.output}")
        print("Expected output: [16]")
        print(f"Test passed: {state.output == [16]}")
    except EmulatorError as e:
        print(f"Emulator error: {e}")

    # Test control flow
    print("\n" + "=" * 50)
    print("Testing control flow: if (10 > 5) result = 1 else result = 0")
    print("=" * 50)

    test_if = """
        LOAD R0, #10
        LOAD R1, #5
        CMP R0, R1
        JGT then_branch
        LOAD R2, #0
        JMP end_if
    then_branch:
        LOAD R2, #1
    end_if:
        STORE R2, [0]
        PRINT R2
        HALT
    """

    try:
        state = run_assembly(test_if)
        print(f"Output: {state.output}")
        print("Expected: [1]")
        print(f"Test passed: {state.output == [1]}")
    except EmulatorError as e:
        print(f"Emulator error: {e}")
