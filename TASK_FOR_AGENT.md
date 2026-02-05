# Task for AI Agent

## Objective

Implement a compiler for the **SimpleCalc** language that:

1. Parses source code into an AST
2. Generates SimpleVM assembly from the AST

## Your Task

Implement two functions:

### 1. `POC/src/parser.py` - Implement `parse(source_code: str) -> dict`

- Input: Source code string (SimpleCalc language)
- Output: AST as a Python dictionary
- Grammar: Defined in `task/grammar.ebnf`

### 2. `POC/src/code_generator.py` - Implement `generate(ast: dict) -> str`

- Input: AST dictionary from the parser
- Output: Assembly code string for SimpleVM
- Instruction Set: Defined in `task/instruction-set.txt`

## Constraints

- Use only the instructions defined in `instruction-set.txt`
- Variables should be stored at memory addresses 0, 1, 2, ... (in order of first appearance)
- All programs must end with `HALT`
- Registers R0-R3 are available for computation

## Testing

Run the test suite to verify your implementation:

```bash
python -m POC.verify.runner
```

All 6 tests should pass when your implementation is correct.

## Example

**Input Source (test_001):**

```
x = 3 + 5;
print(x);
```

**Expected AST:**

```json
{
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
```

**Expected Assembly:**

```asm
LOAD R0, #3
LOAD R1, #5
ADD R0, R0, R1
STORE R0, [0]      ; x stored at address 0
LOAD R0, [0]       ; load x for print
PRINT R0
HALT
```

**Expected State:**

- `memory[0] = 8` (variable x)
- `output = [8]`

## Files Reference

```
POC/
├── task/
│   ├── grammar.ebnf          # Language grammar (READ THIS)
│   └── instruction-set.txt   # Target machine instructions (READ THIS)
├── tests/
│   ├── test_*.src            # Test programs
│   └── test_*.expected       # Expected results
├── src/
│   ├── parser.py             # IMPLEMENT THIS
│   └── code_generator.py     # IMPLEMENT THIS
└── verify/
    ├── emulator.py           # Executes assembly (provided)
    └── runner.py             # Runs tests (provided)
```
