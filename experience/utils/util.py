import avalanche
class SGNaive(avalanche.training.Naive):
    """There are only a couple of modifications needed to
    use Image Segementation:
    - we add a bunch of attributes corresponding to the batch items,
        redefining mb_x and mb_y too
    - _unpack_minibatch sends the dictionary values to the GPU device
    - forward and criterion are adapted for machine translation tasks.
    """

    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch["x"].to("cuda:0")

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch["targets"].to("cuda:0")

    @property
    def mb_task_id(self):
        return self.mbatch["task_labels"]

    def _unpack_minibatch(self):
        # print(self.mbatch)
        pass

