"""Main application entry point for the Self-Healing AI Agent Runtime."""

from shar import Task, TaskRunner


def sample_work(x: int, y: int) -> int:
    """A simple baseline task callable."""
    return x + y


def main() -> None:
    """Demonstrates a simple Task execution using TaskRunner."""
    with TaskRunner() as runner:
        task = Task(
            name="sample_addition",
            callable=sample_work,
            args=(10, 20),
            validator=lambda val: val > 0,
        )
        result = runner.run(task)
        print(
            f"Self-Healing Agent Runtime started. "
            f"Task '{result.task_name}' status: {result.outcome.value}, output: {result.output}"
        )


if __name__ == "__main__":
    main()
