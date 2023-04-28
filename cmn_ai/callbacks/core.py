import re


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

    order = 0

    def set_learner(self, learner):
        self.learner = learner

    def __getattr__(self, k):
        return getattr(self.learner, k)

    @property
    def name(self):
        """
        Returns the name of the callback after removing the word `callback`
        and then convert it to snake (split words by underscores).
        """
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return Callback.camel2snake(name or "callback")

    def __call__(self, event_nm):
        method = getattr(self, event_nm, None)
        if method is not None:
            method()

    @staticmethod
    def camel2snake(name):
        """
        Convert name of callback by inserting underscores between small and capital
        letters. For example, `TestCallback` becomes `test_callback`.
        """
        pattern1 = re.compile("(.)([A-Z][a-z]+)")
        pattern2 = re.compile("([a-z0-9])([A-Z])")
        name = re.sub(pattern1, r"\1_\2", name)
        return re.sub(pattern2, r"\1_\2", name).lower()
