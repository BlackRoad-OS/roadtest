"""
RoadTest - Testing Framework for BlackRoad
Unit testing, integration testing, and test fixtures with assertions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
import asyncio
import inspect
import logging
import time
import traceback

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a test execution."""
    name: str
    status: TestStatus
    duration_ms: float = 0
    error: Optional[str] = None
    traceback: Optional[str] = None
    assertions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_duration_ms: float = 0
    started_at: datetime = field(default_factory=datetime.now)


class AssertionError(Exception):
    """Test assertion failed."""
    pass


class Assert:
    """Assertion helpers."""

    @staticmethod
    def equal(actual: Any, expected: Any, message: str = "") -> None:
        if actual != expected:
            raise AssertionError(
                f"{message or 'Values not equal'}: {actual!r} != {expected!r}"
            )

    @staticmethod
    def not_equal(actual: Any, expected: Any, message: str = "") -> None:
        if actual == expected:
            raise AssertionError(
                f"{message or 'Values should not be equal'}: {actual!r} == {expected!r}"
            )

    @staticmethod
    def true(value: bool, message: str = "") -> None:
        if not value:
            raise AssertionError(message or f"Expected True, got {value!r}")

    @staticmethod
    def false(value: bool, message: str = "") -> None:
        if value:
            raise AssertionError(message or f"Expected False, got {value!r}")

    @staticmethod
    def is_none(value: Any, message: str = "") -> None:
        if value is not None:
            raise AssertionError(message or f"Expected None, got {value!r}")

    @staticmethod
    def is_not_none(value: Any, message: str = "") -> None:
        if value is None:
            raise AssertionError(message or "Expected non-None value")

    @staticmethod
    def is_instance(obj: Any, cls: Type, message: str = "") -> None:
        if not isinstance(obj, cls):
            raise AssertionError(
                message or f"Expected instance of {cls.__name__}, got {type(obj).__name__}"
            )

    @staticmethod
    def contains(container: Any, item: Any, message: str = "") -> None:
        if item not in container:
            raise AssertionError(
                message or f"{item!r} not found in {container!r}"
            )

    @staticmethod
    def raises(exception_type: Type[Exception], func: Callable, *args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except exception_type:
            return
        except Exception as e:
            raise AssertionError(
                f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}"
            )
        raise AssertionError(f"Expected {exception_type.__name__} to be raised")

    @staticmethod
    def length(container: Any, expected: int, message: str = "") -> None:
        actual = len(container)
        if actual != expected:
            raise AssertionError(
                message or f"Expected length {expected}, got {actual}"
            )

    @staticmethod
    def greater_than(actual: Any, expected: Any, message: str = "") -> None:
        if not actual > expected:
            raise AssertionError(
                message or f"Expected {actual!r} > {expected!r}"
            )

    @staticmethod
    def less_than(actual: Any, expected: Any, message: str = "") -> None:
        if not actual < expected:
            raise AssertionError(
                message or f"Expected {actual!r} < {expected!r}"
            )


class Fixture:
    """Test fixture for setup and teardown."""

    def __init__(self, scope: str = "function"):
        self.scope = scope  # function, class, module, session
        self._instances: Dict[str, Any] = {}

    def setup(self) -> Any:
        """Override for setup logic."""
        pass

    def teardown(self, instance: Any) -> None:
        """Override for teardown logic."""
        pass

    def get(self, key: str) -> Any:
        """Get or create fixture instance."""
        if key not in self._instances:
            self._instances[key] = self.setup()
        return self._instances[key]

    def cleanup(self, key: str = None) -> None:
        """Clean up fixture instances."""
        if key and key in self._instances:
            self.teardown(self._instances[key])
            del self._instances[key]
        elif not key:
            for k, v in list(self._instances.items()):
                self.teardown(v)
            self._instances.clear()


class TestCase:
    """Base test case class."""

    def setup(self) -> None:
        """Called before each test."""
        pass

    def teardown(self) -> None:
        """Called after each test."""
        pass

    def setup_class(self) -> None:
        """Called before the test class."""
        pass

    def teardown_class(self) -> None:
        """Called after the test class."""
        pass


class TestRunner:
    """Run tests and collect results."""

    def __init__(self):
        self.fixtures: Dict[str, Fixture] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "before_test": [],
            "after_test": [],
            "before_suite": [],
            "after_suite": []
        }

    def add_hook(self, event: str, handler: Callable) -> None:
        if event in self._hooks:
            self._hooks[event].append(handler)

    def register_fixture(self, name: str, fixture: Fixture) -> None:
        self.fixtures[name] = fixture

    def _get_test_methods(self, test_class: Type) -> List[str]:
        """Get all test methods from a class."""
        return [
            name for name in dir(test_class)
            if name.startswith("test_") and callable(getattr(test_class, name))
        ]

    def _run_hooks(self, event: str, *args) -> None:
        for handler in self._hooks.get(event, []):
            try:
                handler(*args)
            except Exception as e:
                logger.error(f"Hook error: {e}")

    async def run_test(
        self,
        test_instance: TestCase,
        method_name: str
    ) -> TestResult:
        """Run a single test."""
        result = TestResult(name=method_name, status=TestStatus.RUNNING)
        start_time = time.time()

        try:
            self._run_hooks("before_test", test_instance, method_name)
            
            # Setup
            test_instance.setup()

            # Get method
            method = getattr(test_instance, method_name)

            # Check for skip decorator
            if getattr(method, "_skip", False):
                result.status = TestStatus.SKIPPED
                result.metadata["skip_reason"] = getattr(method, "_skip_reason", "")
                return result

            # Run test
            if asyncio.iscoroutinefunction(method):
                await method()
            else:
                method()

            result.status = TestStatus.PASSED

        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            result.traceback = traceback.format_exc()

        finally:
            try:
                test_instance.teardown()
            except Exception as e:
                logger.error(f"Teardown error: {e}")

            result.duration_ms = (time.time() - start_time) * 1000
            self._run_hooks("after_test", test_instance, method_name, result)

        return result

    async def run_class(self, test_class: Type[TestCase]) -> TestSuiteResult:
        """Run all tests in a class."""
        suite_result = TestSuiteResult(name=test_class.__name__)
        
        self._run_hooks("before_suite", test_class)

        test_instance = test_class()
        
        try:
            test_instance.setup_class()
        except Exception as e:
            logger.error(f"Setup class error: {e}")

        for method_name in self._get_test_methods(test_class):
            result = await self.run_test(test_instance, method_name)
            suite_result.results.append(result)

            if result.status == TestStatus.PASSED:
                suite_result.passed += 1
            elif result.status == TestStatus.FAILED:
                suite_result.failed += 1
            elif result.status == TestStatus.SKIPPED:
                suite_result.skipped += 1
            else:
                suite_result.errors += 1

            suite_result.total_duration_ms += result.duration_ms

        try:
            test_instance.teardown_class()
        except Exception as e:
            logger.error(f"Teardown class error: {e}")

        self._run_hooks("after_suite", test_class, suite_result)

        return suite_result


# Decorators
def skip(reason: str = ""):
    """Skip a test."""
    def decorator(func):
        func._skip = True
        func._skip_reason = reason
        return func
    return decorator


def skip_if(condition: bool, reason: str = ""):
    """Conditionally skip a test."""
    def decorator(func):
        if condition:
            func._skip = True
            func._skip_reason = reason
        return func
    return decorator


def timeout(seconds: float):
    """Set timeout for a test."""
    def decorator(func):
        func._timeout = seconds
        return func
    return decorator


def parametrize(params: List[Dict[str, Any]]):
    """Parametrize a test."""
    def decorator(func):
        func._params = params
        return func
    return decorator


class TestManager:
    """High-level test management."""

    def __init__(self):
        self.runner = TestRunner()
        self.suites: List[Type[TestCase]] = []

    def register(self, test_class: Type[TestCase]) -> None:
        """Register a test class."""
        self.suites.append(test_class)

    def register_fixture(self, name: str, setup: Callable, teardown: Callable = None) -> None:
        """Register a fixture."""
        class DynamicFixture(Fixture):
            def setup(self):
                return setup()
            def teardown(self, instance):
                if teardown:
                    teardown(instance)
        
        self.runner.register_fixture(name, DynamicFixture())

    async def run_all(self) -> List[TestSuiteResult]:
        """Run all registered tests."""
        results = []
        for suite in self.suites:
            result = await self.runner.run_class(suite)
            results.append(result)
        return results

    def print_results(self, results: List[TestSuiteResult]) -> None:
        """Print test results."""
        total_passed = sum(r.passed for r in results)
        total_failed = sum(r.failed for r in results)
        total_skipped = sum(r.skipped for r in results)
        total_errors = sum(r.errors for r in results)

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        for suite in results:
            status_symbol = "✓" if suite.failed == 0 and suite.errors == 0 else "✗"
            print(f"\n{status_symbol} {suite.name}")
            
            for test in suite.results:
                if test.status == TestStatus.PASSED:
                    symbol = "  ✓"
                elif test.status == TestStatus.FAILED:
                    symbol = "  ✗"
                elif test.status == TestStatus.SKIPPED:
                    symbol = "  ⊘"
                else:
                    symbol = "  !"
                
                print(f"{symbol} {test.name} ({test.duration_ms:.2f}ms)")
                
                if test.error:
                    print(f"      Error: {test.error}")

        print("\n" + "-" * 60)
        print(f"Passed: {total_passed} | Failed: {total_failed} | Skipped: {total_skipped} | Errors: {total_errors}")
        print(f"Total time: {sum(r.total_duration_ms for r in results):.2f}ms")


# Example usage
async def example_usage():
    """Example test framework usage."""
    manager = TestManager()

    class UserTests(TestCase):
        def setup(self):
            self.users = []

        def teardown(self):
            self.users.clear()

        def test_create_user(self):
            user = {"id": 1, "name": "Alice"}
            self.users.append(user)
            Assert.equal(len(self.users), 1)
            Assert.equal(self.users[0]["name"], "Alice")

        def test_user_validation(self):
            Assert.raises(KeyError, lambda: {}["missing"])

        @skip("Not implemented yet")
        def test_delete_user(self):
            pass

        def test_user_list(self):
            self.users.extend([{"id": 1}, {"id": 2}])
            Assert.length(self.users, 2)
            Assert.contains([u["id"] for u in self.users], 1)

    manager.register(UserTests)

    results = await manager.run_all()
    manager.print_results(results)
