# DSL Compiler Benchmark - Proof of Concept

This POC demonstrates the core idea of the DSL Compiler Benchmark:

**Task for AI Agent:** Given an EBNF grammar and an instruction set for a toy programming language → write a compiler (parser + code generator) for this language.

## Structure

```
(project root)/
├── tests/
│   ├── test_001.src          # Test program source files
│   ├── test_001.expected     # Expected final emulator state
│   └── ...
├── workspace/
│   ├── parser.py             # [TO IMPLEMENT] Parser: source → AST
│   ├── code_generator.py     # [TO IMPLEMENT] Code gen: AST → Assembly
│   └── task/
│       ├── grammar.ebnf          # EBNF grammar defining the language syntax
│       └── instruction-set.txt   # Target architecture instruction set
└── verify/
    ├── emulator.py           # Emulator to execute assembly
    └── runner.py             # Test runner to verify correctness
```

## How It Works

1. **Task Definition**: The `workspace/task/` folder defines the language (grammar) and target machine (instruction set)
2. **Test Cases**: The `tests/` folder contains programs in the toy language with expected outputs
3. **Agent Implements**: The agent must implement `parser.py` and `code_generator.py`
4. **Verification**: The `verify/` folder runs the compiled code and checks against expected state

## Running the POC

```bash
# After implementing parser.py and code_generator.py:
python -m verify.runner
```

## Language Overview (This POC)

This POC uses a simple arithmetic language with:

- Integer literals and variables
- Arithmetic operations: `+`, `-`, `*`, `/`
- Assignment statements
- Print statements
- Simple if-else control flow

Target: Simple register-based machine with 4 registers (R0-R3), memory, and basic arithmetic instructions.

## Hard mode

- Removes some helper comments, can just use `git apply hard.patch`

## Reference Solution

- Sample solution: in `solution.patch`

## Summary of POC

In cursor:

- current environment : Opus 4.5 and Codex 5.2 extra high pass
- Hard mode: Sonnet 4.5 fails, Opus 4.5 passes
