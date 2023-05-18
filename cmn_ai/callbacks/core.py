"""
The main module that has the definition of the `Callback` base class as
well as all the Exceptions that a callback may raise.
"""
import re
from typing import Any


class CancelFitException(Exception):
    """Stop training and move to after_fit."""


class CancelEpochException(Exception):
    """Stop current epoch and move to after_epoch."""


class CancelTrainException(Exception):
    """Stop training current epoch and move to after_train."""


class CancelValidException(Exception):
    """Stop validation phase and move after_validate."""


class CancelBatchException(Exception):
    """Stop current batch and move to after_batch."""


class CancelStepException(Exception):
    """Skip stepping the optimizer."""


class CancelBackwardException(Exception):
    """Skip the backward pass and move to after_backward."""


class Callback:
    """Base class for all callbacks."""

    order: int = 0

    def set_learner(self, learner) -> None:
        """
        Set the learner as an attribute so that callbacks can access learner's
        attributes without the need to pass `learner` for every single method
        in every callback.

        Parameters
        ----------
        learner : Learner
            Learner that the callback will be called when some events happens.
        """
        self.learner = learner

    def __getattr__(self, k) -> Any:
        """
        This would allow us to use `self.obj` instead of `self.learner.obj`
        when we know `obj` is in learner because it will only be called when
        `getattribute` returns `AttributeError`.
        """
        return getattr(self.learner, k)

    @property
    def name(self) -> str:
        """
        Returns the name of the callback after removing the word `callback`
        and then convert it to snake (split words by underscores).
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return Callback.camel2snake(name or "callback")

    def __call__(self, event_nm: str) -> Any | None:
        method = getattr(self, event_nm, None)
        if method is not None:
            method()

    @staticmethod
    def camel2snake(name: str) -> str:
        """
        Convert name of callback by inserting underscores between small and capital
        letters. For example, `TestCallback` becomes `test_callback`.
        """
        pattern1 = re.compile("(.)([A-Z][a-z]+)")
        pattern2 = re.compile("([a-z0-9])([A-Z])")
        name = re.sub(pattern1, r"\1_\2", name)
        return re.sub(pattern2, r"\1_\2", name).lower()
