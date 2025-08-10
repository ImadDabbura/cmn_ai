"""
Core callback system for training loop management.

This module provides the foundational callback system that allows for
custom behavior injection into machine learning training loops. It defines
the base `Callback` class and a set of control flow exceptions that
enable fine-grained control over the training process.

Classes
-------
Callback
    Base class for all callbacks, providing the interface for training
    loop integration and utility methods for callback management.
CancelFitException
    Exception to stop training and move to after_fit phase.
CancelEpochException
    Exception to stop current epoch and move to after_epoch phase.
CancelTrainException
    Exception to stop training phase and move to after_train phase.
CancelValidateException
    Exception to stop validation phase and move to after_validate phase.
CancelBatchException
    Exception to stop current batch and move to after_batch phase.
CancelStepException
    Exception to skip optimizer step and move to after_step phase.
CancelBackwardException
    Exception to skip backward pass and move to after_backward phase.

Notes
-----
The callback system works by defining specific event names that correspond
to different phases of the training loop. Callbacks can implement methods
with these event names to be called at the appropriate times:

- `before_fit`: Called before training begins
- `after_fit`: Called after training completes
- `before_epoch`: Called before each epoch
- `after_epoch`: Called after each epoch
- `before_train`: Called before training phase of each epoch
- `after_train`: Called after training phase of each epoch
- `before_validate`: Called before validation phase of each epoch
- `after_validate`: Called after validation phase of each epoch
- `before_batch`: Called before processing each batch
- `after_batch`: Called after processing each batch
- `before_step`: Called before optimizer step
- `after_step`: Called after optimizer step
- `before_backward`: Called before backward pass
- `after_backward`: Called after backward pass

Examples
--------
Creating a custom callback:

>>> class MyCallback(Callback):
...     def before_epoch(self):
...         print(f"Starting epoch {self.epoch}")
...
...     def after_batch(self):
...         if self.loss > 0.5:
...             raise CancelEpochException("Loss too high")

Using control flow exceptions:

>>> class EarlyStoppingCallback(Callback):
...     def after_epoch(self):
...         if self.epoch > 10 and self.valid_loss > self.best_loss:
...             raise CancelFitException("Early stopping triggered")
"""

import re
from typing import Any


class CancelFitException(Exception):
    """
    Exception raised to stop training and move to `after_fit`.

    This exception is used to interrupt the training process and
    immediately proceed to the after_fit phase of the training loop.
    """

    pass


class CancelEpochException(Exception):
    """
    Exception raised to stop current epoch and move to `after_epoch`.

    This exception is used to interrupt the current epoch and
    immediately proceed to the after_epoch phase.
    """

    pass


class CancelTrainException(Exception):
    """
    Exception raised to stop training current epoch and move to `after_train`.

    This exception is used to interrupt the training phase of the current
    epoch and immediately proceed to the after_train phase.
    """

    pass


class CancelValidateException(Exception):
    """
    Exception raised to stop validation phase and move to `after_validate`.

    This exception is used to interrupt the validation phase and
    immediately proceed to the after_validate phase.
    """

    pass


class CancelBatchException(Exception):
    """
    Exception raised to stop current batch and move to `after_batch`.

    This exception is used to interrupt the current batch processing
    and immediately proceed to the after_batch phase.
    """

    pass


class CancelStepException(Exception):
    """
    Exception raised to skip stepping the optimizer and move to `after_step`.

    This exception is used to skip the optimizer step and immediately
    proceed to the after_step phase.
    """

    pass


class CancelBackwardException(Exception):
    """
    Exception raised to skip the backward pass and move to `after_backward`.

    This exception is used to skip the backward pass computation and
    immediately proceed to the after_backward phase.
    """

    pass


class Callback:
    """
    Base class for all callbacks.

    A callback is a mechanism to inject custom behavior into the training
    loop at specific points. Callbacks can be used for logging, early
    stopping, learning rate scheduling, and other custom functionality.

    Attributes
    ----------
    order : int
        The order in which callbacks should be executed. Lower numbers
        are executed first. Default is 0.
    learner : Any
        Reference to the learner object, set by `set_learner()`.

    Notes
    -----
    Subclasses should implement specific callback methods that correspond
    to training events (e.g., `before_fit`, `after_epoch`, etc.).
    """

    order: int = 0

    def __init__(self):
        """Initialize the callback."""
        self.learner = None

    def set_learner(self, learner: Any) -> None:
        """
        Set the learner as an attribute so that callbacks can access
        learner's attributes without the need to pass `learner` for
        every single method in every callback.

        Parameters
        ----------
        learner : Any
            Learner that the callback will be called when some events
            happens. This object will be stored as `self.learner`.
        """
        self.learner = learner

    def __getattr__(self, k: str) -> Any:
        """
        Allow access to learner attributes directly through the callback.

        This would allow us to use `self.obj` instead of
        `self.learner.obj` when we know `obj` is in learner because it
        will only be called when `getattribute` returns `AttributeError`.

        Parameters
        ----------
        k : str
            The attribute name to access from the learner.

        Returns
        -------
        Any
            The attribute value from the learner object.

        Raises
        ------
        AttributeError
            If the attribute is not found in the learner object.
        """
        try:
            learner = getattr(self, "learner")
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{k}'"
            )

        if learner is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{k}'"
            )
        return getattr(learner, k)

    @property
    def name(self) -> str:
        """
        Returns the name of the callback after removing the word
        `callback` and then convert it to snake (split words by
        underscores).

        Returns
        -------
        str
            The callback name in snake_case format with 'Callback' suffix removed.
            For example, 'TestCallback' becomes 'test'.
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return Callback.camel2snake(name or "callback")

    def __call__(self, event_nm: str) -> Any:
        """
        Call the callback method corresponding to the given event name.

        If the callback has a method with the same name as the event,
        it will be called. Otherwise, nothing happens.

        Parameters
        ----------
        event_nm : str
            The name of the event to handle (e.g., 'before_fit', 'after_epoch').

        Returns
        -------
        Any
            The return value of the callback method, or None if the method
            doesn't exist.
        """
        method = getattr(self, event_nm, None)
        if method is not None:
            return method()
        return None

    @staticmethod
    def camel2snake(name: str) -> str:
        """
        Convert camelCase name to snake_case by inserting underscores.

        Inserts underscores between lowercase and uppercase letters.
        For example, `TestCallback` becomes `test_callback`.

        Parameters
        ----------
        name : str
            The camelCase string to convert.

        Returns
        -------
        str
            The converted snake_case string.

        Examples
        --------
        >>> Callback.camel2snake("TestCallback")
        'test_callback'
        >>> Callback.camel2snake("MyCustomCallback")
        'my_custom_callback'
        """
        pattern1 = re.compile("(.)([A-Z][a-z]+)")
        pattern2 = re.compile("([a-z0-9])([A-Z])")
        name = re.sub(pattern1, r"\1_\2", name)
        return re.sub(pattern2, r"\1_\2", name).lower()
