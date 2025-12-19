"""
Workflow Base Classes

Provides structured execution of multi-step agent tasks with verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow"""
    name: str
    description: str
    action: Callable[..., Any]
    verify: Optional[Callable[..., bool]] = None
    required: bool = True
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    success: bool
    steps_completed: int
    steps_total: int
    output_path: Optional[Path] = None
    data: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"WorkflowResult({status}, {self.steps_completed}/{self.steps_total} steps)"


class Workflow(ABC):
    """
    Base class for structured, multi-step workflows.

    Provides:
    - Step-by-step execution with logging
    - Verification after each step
    - Error handling and recovery
    - Result tracking
    """

    def __init__(self, name: str, output_dir: Optional[Path] = None):
        self.name = name
        self.output_dir = output_dir or Path("./output")
        self.steps: list[WorkflowStep] = []
        self.context: dict[str, Any] = {}
        self._setup_steps()

    @abstractmethod
    def _setup_steps(self) -> None:
        """Define workflow steps. Override in subclass."""
        pass

    def add_step(
        self,
        name: str,
        description: str,
        action: Callable[..., Any],
        verify: Optional[Callable[..., bool]] = None,
        required: bool = True
    ) -> None:
        """Add a step to the workflow"""
        self.steps.append(WorkflowStep(
            name=name,
            description=description,
            action=action,
            verify=verify,
            required=required
        ))

    async def run(self, **kwargs) -> WorkflowResult:
        """Execute all workflow steps"""
        self.context.update(kwargs)
        completed = 0
        errors = []

        logger.info(f"Starting workflow: {self.name}")

        for i, step in enumerate(self.steps):
            logger.info(f"Step {i+1}/{len(self.steps)}: {step.name}")
            step.status = StepStatus.RUNNING

            try:
                # Execute step
                step.result = await self._run_step(step)

                # Verify if verification function provided
                if step.verify:
                    if not step.verify(step.result, self.context):
                        raise ValueError(f"Verification failed for step: {step.name}")

                step.status = StepStatus.COMPLETED
                completed += 1
                logger.info(f"  ✓ {step.name} completed")

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                errors.append(f"{step.name}: {e}")
                logger.error(f"  ✗ {step.name} failed: {e}")

                if step.required:
                    logger.error(f"Required step failed, stopping workflow")
                    break

        success = completed == len(self.steps) or all(
            s.status == StepStatus.COMPLETED or not s.required
            for s in self.steps
        )

        return WorkflowResult(
            success=success,
            steps_completed=completed,
            steps_total=len(self.steps),
            output_path=self.context.get("output_path"),
            data=self.context,
            errors=errors
        )

    async def _run_step(self, step: WorkflowStep) -> Any:
        """Run a single step, handling both sync and async actions"""
        import asyncio

        result = step.action(self.context)
        if asyncio.iscoroutine(result):
            result = await result

        # Store result in context for subsequent steps
        self.context[f"{step.name}_result"] = result
        return result
