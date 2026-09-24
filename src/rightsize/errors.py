"""Exceptions shared across the package. Standard library only."""

from __future__ import annotations


class RightsizeError(Exception):
    """Base class for all Rightsize errors."""


class NotImplementedYet(RightsizeError):
    """Raised by public functions whose feature has not landed yet."""

    def __init__(self, feature: str, plan: str) -> None:
        self.feature = feature
        self.plan = plan
        super().__init__(
            f"rightsize.{feature} is not implemented yet in this placeholder release. "
            f"Plan and TODO: {plan}"
        )


class MissingExtraError(RightsizeError):
    """Raised when an optional dependency group is needed but not installed.

    Lightweight rule 4: never an ImportError at startup, always a clear install line.
    """

    def __init__(self, extra: str, purpose: str) -> None:
        self.extra = extra
        self.purpose = purpose
        super().__init__(
            f"{purpose} needs the optional extra '{extra}'. "
            f'Install it with:  pip install "rightsize[{extra}]"'
        )
