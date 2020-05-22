from assignment.utils import set_np_printoptions
from assignment.model import (
    SoftEmbeddedDecisionRules as SoftRules,
    HardEmbeddedDecisionRules as HardRules
)
import numpy as np


__all__ = names = (
    'Noop', 'ConfusionMatrix', 'ConfusionMatrixJointNodes',
    'IgnoredSamples', 'HardEmbeddedDecisionRules', 'SoftEmbeddedDecisionRules')
keys = ('path_graph', 'path_wnids', 'classes', 'dataset')


def add_arguments(parser):
    pass


class Noop:

    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self, classes=()):
        set_np_printoptions()

        self.classes = classes
        self.num_classes = len(classes)
        self.epoch = None

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_train(self, epoch):
        assert epoch == self.epoch

    def update_batch(self, outputs, targets):
        pass

    def end_train(self, epoch):
        assert epoch == self.epoch

    def start_test(self, epoch):
        assert epoch == self.epoch

    def end_test(self, epoch):
        assert epoch == self.epoch

    def end_epoch(self, epoch):
        assert epoch == self.epoch


class ConfusionMatrix(Noop):

    def __init__(self, classes):
        super().__init__(classes)
        self.k = len(classes)
        self.m = None

    def start_train(self, epoch):
        super().start_train(epoch)
        raise NotImplementedError()

    def start_test(self, epoch):
        super().start_test(epoch)
        self.m = np.zeros((self.k, self.k))

    def update_batch(self, outputs, targets):
        super().update_batch(outputs, targets)
        _, predicted = outputs.max(1)
        if len(predicted.shape) == 1:
            predicted = predicted.numpy().ravel()
            targets = targets.numpy().ravel()
            ConfusionMatrix.update(self.m, predicted, targets)

    def end_test(self, epoch):
        super().end_test(epoch)
        recall = self.recall()
        for row, cls in zip(recall, self.classes):
            print(row, cls)
        print(recall.diagonal(), '(diagonal)')

    @staticmethod
    def update(confusion_matrix, preds, labels):
        preds = tuple(preds)
        labels = tuple(labels)

        for pred, label in zip(preds, labels):
            confusion_matrix[label, pred] += 1

    @staticmethod
    def normalize(confusion_matrix, axis):
        total = confusion_matrix.astype(np.float).sum(axis=axis)
        total = total[:, None] if axis == 1 else total[None]
        return confusion_matrix / total

    def recall(self):
        return ConfusionMatrix.normalize(self.m, 1)

    def precision(self):
        return ConfusionMatrix.normalize(self.m, 0)


class IgnoredSamples(Noop):
    """ Counter for number of ignored samples in decision tree """

    def __init__(self, classes=()):
        super().__init__(classes)
        self.ignored = None

    def start_test(self, epoch):
        super().start_test(epoch)
        self.ignored = 0

    def update_batch(self, outputs, targets):
        super().update_batch(outputs, targets)
        self.ignored += outputs[:,0].eq(-1).sum().item()

    def end_test(self, epoch):
        super().end_test(epoch)
        print("Ignored Samples: {}".format(self.ignored))


class HardEmbeddedDecisionRules(Noop):
    """Evaluation is hard."""

    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True

    name = 'NBDT-Hard'

    def __init__(self, *args, Rules=HardRules, **kwargs):
        self.rules = Rules(*args, **kwargs)

    def update_batch(self, outputs, targets):
        super().update_batch(outputs, targets)
        predicted = self.rules.forward(outputs).max(1)[1].to(targets.device)
        _,decisions = self.rules.forward_with_decisions(outputs)
        n_samples = outputs.size(0)
        self.rules.total += n_samples
        self.rules.correct += (predicted == targets).sum().item()
        accuracy = round(self.rules.correct / float(self.rules.total), 4) * 100
        return predicted,decisions

    def end_test(self, epoch):
        super().end_test(epoch)
        accuracy = round(self.rules.correct / self.rules.total * 100., 2)
        print(f'{self.name} Accuracy: {accuracy}%, {self.rules.correct}/{self.rules.total}')


class SoftEmbeddedDecisionRules(HardEmbeddedDecisionRules):
    """Evaluation is soft."""

    name = 'NBDT-Soft'

    def __init__(self, *args, Rules=None, **kwargs):
        super().__init__(*args, Rules=SoftRules, **kwargs)
