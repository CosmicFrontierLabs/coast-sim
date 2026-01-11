"""Command queue for ACS state machine.

This module provides a CommandQueue class that manages the queue of pending
ACS commands and the history of executed commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..common import ACSCommandType

if TYPE_CHECKING:
    from .acs_command import ACSCommand


class CommandQueue:
    """Priority queue for ACS commands sorted by execution time.

    Manages pending commands and executed command history. Commands are
    automatically sorted by execution time when enqueued.
    """

    def __init__(self) -> None:
        """Initialize empty command queue."""
        self._pending: list[ACSCommand] = []
        self._executed: list[ACSCommand] = []

    @property
    def pending(self) -> list[ACSCommand]:
        """Return list of pending commands (read-only view)."""
        return list(self._pending)

    @property
    def executed(self) -> list[ACSCommand]:
        """Return list of executed commands (read-only view)."""
        return list(self._executed)

    @property
    def is_empty(self) -> bool:
        """Return True if no pending commands."""
        return len(self._pending) == 0

    def enqueue(self, command: ACSCommand) -> None:
        """Add a command to the queue, maintaining time order.

        Args:
            command: The command to enqueue.
        """
        self._pending.append(command)
        self._pending.sort(key=lambda cmd: cmd.execution_time)

    def pop_due(self, utime: float) -> ACSCommand | None:
        """Pop the next command due for execution, if any.

        Args:
            utime: Current time in unix seconds.

        Returns:
            The next due command, or None if no commands are due.
        """
        if self._pending and self._pending[0].execution_time <= utime:
            return self._pending.pop(0)
        return None

    def mark_executed(self, command: ACSCommand) -> None:
        """Record a command as executed.

        Args:
            command: The command that was executed.
        """
        self._executed.append(command)

    def clear(self) -> None:
        """Clear all pending commands (keeps executed history)."""
        self._pending.clear()

    def clear_all(self) -> None:
        """Clear both pending and executed commands."""
        self._pending.clear()
        self._executed.clear()

    def filter_pending(self, command_type: ACSCommandType) -> list[ACSCommand]:
        """Return pending commands of a specific type.

        Args:
            command_type: The type of commands to filter for.

        Returns:
            List of matching pending commands.
        """
        return [c for c in self._pending if c.command_type == command_type]

    def remove_pending_type(self, command_type: ACSCommandType) -> int:
        """Remove all pending commands of a specific type.

        Args:
            command_type: The type of commands to remove.

        Returns:
            Number of commands removed.
        """
        original_len = len(self._pending)
        self._pending = [c for c in self._pending if c.command_type != command_type]
        return original_len - len(self._pending)

    def has_pending_type(self, command_type: ACSCommandType) -> bool:
        """Check if any pending commands of a specific type exist.

        Args:
            command_type: The type to check for.

        Returns:
            True if at least one command of the type is pending.
        """
        return any(c.command_type == command_type for c in self._pending)

    def next_execution_time(self) -> float | None:
        """Return execution time of next pending command, if any.

        Returns:
            Execution time of next command, or None if queue is empty.
        """
        if self._pending:
            return self._pending[0].execution_time
        return None

    def __len__(self) -> int:
        """Return number of pending commands."""
        return len(self._pending)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CommandQueue(pending={len(self._pending)}, "
            f"executed={len(self._executed)})"
        )
