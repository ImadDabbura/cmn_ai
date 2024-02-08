import fastcore.all as fc

from ..callbacks.training import SingleBatchCallback
from ..learner import Learner
from ..plot import show_images
from ..utils.utils import listify


class VisionLearner(Learner):
    """
    Learner that knows how to handle vision's related training/inference.
    """

    @fc.delegates(show_images)
    def show_batch(self, sample_sz, callbacks=None, **kwargs):
        """
        Show batch of images of size `sample_sz`.

        Parameters
        ----------
        sample_sz : int, default=1
            Number of input samples to show.
        callbacks : Iterable[Callback] | None, default=None
            Callbacks to add to the existing callbacks. The added
            callbacks will be removed  before `show_batch` returns.
        """
        self.fit(1, callbacks=[SingleBatchCallback()] + listify(callbacks))
        show_images(self.xb[0][:sample_sz], **kwargs)
