"""
Test Runner for POC
===================

Runs the compiler on test programs and verifies correctness.

Usage:
    python -m POC.verify.runner

Or from the POC directory:
    python verify/runner.py
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from verify.emulator import EmulatorError, EmulatorState, SimpleVMEmulator

# Add parent directories to path for imports
script_dir = Path(__file__).parent
poc_dir = script_dir.parent
sys.path.insert(0, str(poc_dir))
sys.path.insert(0, str(poc_dir.parent))


@dataclass
class TestResult:
    """Result of running a single test."""
    test_name: str
    passed: bool
    error: str | None = None
    expected_output: list[int] | None = None
    actual_output: list[int] | None = None
    expected_variables: dict[str, int] | None = None
    actual_variables: dict[str, int] | None = None


class TestRunner:
    """
    Runs tests for the POC compiler.

    Flow:
    1. Load source file (.src)
    2. Parse source into AST (using src/parser.py)
    3. Generate assembly from AST (using src/code_generator.py)
    4. Execute assembly with emulator
    5. Compare results with expected (.expected file)
    """

    def __init__(self, poc_dir: Path):
        self.poc_dir = poc_dir
        self.tests_dir = poc_dir / "tests"
        self.emulator = SimpleVMEmulator()

    def discover_tests(self) -> list[str]:
        """Find all test files."""
        tests = []
        for src_file in self.tests_dir.glob("*.src"):
            test_name = src_file.stem
            expected_file = self.tests_dir / f"{test_name}.expected"
            if expected_file.exists():
                tests.append(test_name)
        return sorted(tests)

    def load_test(self, test_name: str) -> tuple[str, dict]:
        """Load source and expected results for a test."""
        src_path = self.tests_dir / f"{test_name}.src"
        expected_path = self.tests_dir / f"{test_name}.expected"
        source = src_path.read_text()
        expected = json.loads(expected_path.read_text())

        return source, expected

    def run_test(self, test_name: str) -> TestResult:
        """Run a single test and return the result."""
        try:
            source, expected = self.load_test(test_name)
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                error=f"Failed to load test: {e}"
            )
        # Try to import the compiler modules
        try:
            from workspace.code_generator import generate
            from workspace.parser import parse
        except ImportError as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                error=f"Failed to import compiler: {e}"
            )
        # Parse source to AST
        try:
            ast = parse(source)
        except NotImplementedError:
            return TestResult(
                test_name=test_name,
                passed=False,
                error="Parser not implemented"
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                error=f"Parse error: {e}"
            )
        # Generate assembly
        try:
            assembly = generate(ast)
        except NotImplementedError:
            return TestResult(
                test_name=test_name,
                passed=False,
                error="Code generator not implemented"
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                error=f"Code generation error: {e}"
            )
        # Execute assembly
        try:
            state = self.emulator.run(assembly)
        except EmulatorError as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                error=f"Emulator error: {e}"
            )
        # Verify results
        return self._verify_results(test_name, state, expected)

    def _verify_results(self, test_name: str, state: EmulatorState, expected: dict) -> TestResult:
        """Verify emulator state against expected results."""
        errors = []
        # Check output
        expected_output = expected.get("output", [])
        if state.output != expected_output:
            errors.append(f"Output mismatch: expected {expected_output}, got {state.output}")
        # Check variables (if symbol table is available)
        expected_vars = expected.get("variables", {})
        actual_vars = {}
        # Extract variables from memory based on order of appearance
        # This requires the code generator to use consistent addressing
        for i, (name, value) in enumerate(expected_vars.items()):
            actual_value = state.memory[i]
            actual_vars[name] = actual_value
            if actual_value != value:
                errors.append(f"Variable '{name}' mismatch: expected {value}, got {actual_value}")

        if errors:
            return TestResult(
                test_name=test_name,
                passed=False,
                error="; ".join(errors),
                expected_output=expected_output,
                actual_output=state.output,
                expected_variables=expected_vars,
                actual_variables=actual_vars
            )

        return TestResult(
            test_name=test_name,
            passed=True,
            expected_output=expected_output,
            actual_output=state.output,
            expected_variables=expected_vars,
            actual_variables=actual_vars
        )

    def run_all_tests(self) -> list[TestResult]:
        """Run all discovered tests."""
        tests = self.discover_tests()
        results = []
        for test_name in tests:
            result = self.run_test(test_name)
            results.append(result)
        return results

    def print_results(self, results: list[TestResult]):
        """Print test results in a readable format."""
        print("=" * 70)
        print("POC Compiler Test Results")
        print("=" * 70)
        passed = 0
        failed = 0
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n{status}: {result.test_name}")
            if result.passed:
                passed += 1
                print(f"  Output: {result.actual_output}")
            else:
                failed += 1
                print(f"  Error: {result.error}")
                if result.expected_output is not None:
                    print(f"  Expected output: {result.expected_output}")
                    print(f"  Actual output:   {result.actual_output}")
        print("\n" + "=" * 70)
        print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
        print("=" * 70)
        return failed == 0


def run_with_assembly(assembly: str, expected: dict, test_name: str = "manual") -> TestResult:
    """
    Run a test with pre-generated assembly (for testing without parser/codegen).

    Useful for verifying the emulator works correctly.
    """
    emulator = SimpleVMEmulator()

    try:
        state = emulator.run(assembly)
    except EmulatorError as e:
        return TestResult(
            test_name=test_name,
            passed=False,
            error=f"Emulator error: {e}"
        )

    # Verify output
    expected_output = expected.get("output", [])
    if state.output != expected_output:
        return TestResult(
            test_name=test_name,
            passed=False,
            error=f"Output mismatch: expected {expected_output}, got {state.output}",
            expected_output=expected_output,
            actual_output=state.output
        )

    return TestResult(
        test_name=test_name,
        passed=True,
        expected_output=expected_output,
        actual_output=state.output
    )


def verify_emulator():
    """Verify the emulator works correctly with hand-written assembly."""
    print("=" * 70)
    print("Verifying Emulator with Hand-Written Assembly")
    print("=" * 70)

    test_cases = [
        # Test 1: Simple addition
        {
            "name": "simple_add",
            "assembly": """
                LOAD R0, #3
                LOAD R1, #5
                ADD R0, R0, R1
                STORE R0, [0]
                PRINT R0
                HALT
            """,
            "expected": {"output": [8], "variables": {"x": 8}}
        },
        # Test 2: Arithmetic with precedence
        {
            "name": "arithmetic_precedence",
            "assembly": """
                ; a = 2 + 3 * 4 = 14
                LOAD R0, #3
                LOAD R1, #4
                MUL R0, R0, R1      ; R0 = 12
                LOAD R1, #2
                ADD R0, R1, R0      ; R0 = 14
                STORE R0, [0]       ; a = 14
                PRINT R0

                ; b = (2 + 3) * 4 = 20
                LOAD R0, #2
                LOAD R1, #3
                ADD R0, R0, R1      ; R0 = 5
                LOAD R1, #4
                MUL R0, R0, R1      ; R0 = 20
                STORE R0, [1]       ; b = 20
                PRINT R0

                ; c = 10 / 3 = 3
                LOAD R0, #10
                LOAD R1, #3
                DIV R0, R0, R1
                STORE R0, [2]       ; c = 3
                PRINT R0

                HALT
            """,
            "expected": {"output": [14, 20, 3]}
        },
        # Test 3: If-else
        {
            "name": "if_else",
            "assembly": """
                ; x = 10, y = 5
                LOAD R0, #10
                STORE R0, [0]       ; x = 10
                LOAD R0, #5
                STORE R0, [1]       ; y = 5

                ; if (x > y) { result = x - y } else { result = y - x }
                LOAD R0, [0]        ; R0 = x
                LOAD R1, [1]        ; R1 = y
                CMP R0, R1
                JGT then_branch
                ; else branch
                SUB R2, R1, R0
                JMP end_if
            then_branch:
                SUB R2, R0, R1
            end_if:
                STORE R2, [2]       ; result = 5
                PRINT R2
                HALT
            """,
            "expected": {"output": [5]}
        },
        # Test 4: Unary minus
        {
            "name": "unary_minus",
            "assembly": """
                ; a = 5
                LOAD R0, #5
                STORE R0, [0]

                ; b = -a = -5
                NEG R1, R0
                STORE R1, [1]
                PRINT R1

                ; c = -10 + 3 = -7
                LOAD R0, #10
                NEG R0, R0
                LOAD R1, #3
                ADD R0, R0, R1
                STORE R0, [2]
                PRINT R0

                ; d = 5 * -2 = -10
                LOAD R0, #5
                LOAD R1, #2
                NEG R1, R1
                MUL R0, R0, R1
                STORE R0, [3]
                PRINT R0

                HALT
            """,
            "expected": {"output": [-5, -7, -10]}
        },
    ]
    all_passed = True
    for test in test_cases:
        result = run_with_assembly(test["assembly"], test["expected"], test["name"])
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{status}: {result.test_name}")
        if not result.passed:
            print(f"  Error: {result.error}")
            all_passed = False
        else:
            print(f"  Output: {result.actual_output}")

    print("\n" + "=" * 70)
    return all_passed


def main():
    """Main entry point."""
    # First verify emulator
    emulator_ok = verify_emulator()
    if not emulator_ok:
        print("\nEmulator verification failed!")
        sys.exit(1)

    print("\n\nEmulator verification passed!")
    print("\nNow running full compiler tests...")
    print("(These will fail until parser and code_generator are implemented)\n")
    # Run compiler tests
    poc_dir = Path(__file__).parent.parent
    runner = TestRunner(poc_dir)
    results = runner.run_all_tests()
    success = runner.print_results(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
