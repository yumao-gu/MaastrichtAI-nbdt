import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from assignment.data.custom import Node #, dataset_to_dummy_classes
from assignment.model import HardEmbeddedDecisionRules, SoftEmbeddedDecisionRules
from assignment.utils import (
    Colors, dataset_to_default_path_graph, dataset_to_default_path_wnids,
    hierarchy_to_path_graph
)

__all__ = names = ('HardTreeSupLoss', 'SoftTreeSupLoss', 'CrossEntropyLoss')
keys = (
    'path_graph', 'path_wnids', 'tree_supervision_weight',
    'classes', 'dataset', 'criterion'
)

def add_arguments(parser):
    parser.add_argument('--hierarchy',
                        help='Hierarchy to use. If supplied, will be used to '
                        'generate --path-graph. --path-graph takes precedence.')
    parser.add_argument('--path-graph', help='Path to graph-*.json file.')  # WARNING: hard-coded suffix -build in generate_fname
    parser.add_argument('--path-wnids', help='Path to wnids.txt file.')
    parser.add_argument('--tree-supervision-weight', type=float, default=1,
                        help='Weight assigned to tree supervision losses')


def set_default_values(args):
    assert not (args.hierarchy and args.path_graph), \
        'Only one, between --hierarchy and --path-graph can be provided.'
    if 'TreeSupLoss' not in args.loss:
        return
    if args.hierarchy and not args.path_graph:
        args.path_graph = hierarchy_to_path_graph(args.dataset, args.hierarchy)
    if not args.path_graph:
        args.path_graph = dataset_to_default_path_graph(args.dataset)
    if not args.path_wnids:
        args.path_wnids = dataset_to_default_path_wnids(args.dataset)


CrossEntropyLoss = nn.CrossEntropyLoss


class TreeSupLoss(nn.Module):

    accepts_criterion = lambda criterion, **kwargs: criterion
    accepts_dataset = lambda trainset, **kwargs: trainset.__class__.__name__
    accepts_path_graph = True
    accepts_path_wnids = True
    accepts_classes = True
    accepts_tree_supervision_weight = True
    accepts_classes = lambda trainset, **kwargs: trainset.classes

    def __init__(self,
            dataset,
            criterion,
            path_graph=None,
            path_wnids=None,
            classes=None,
            hierarchy=None,
            Rules=HardEmbeddedDecisionRules,
            **kwargs):
        super().__init__()

        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            print('MNIST do not support the dummy classes')
            # classes = dataset_to_dummy_classes(dataset)

        self.init(dataset, criterion, path_graph, path_wnids, classes,
            Rules=Rules, **kwargs)

    def init(self,
            dataset,
            criterion,
            path_graph,
            path_wnids,
            classes,
            Rules,
            tree_supervision_weight=1.):
        """
        Extra init method makes clear which arguments are finally necessary for
        this class to function. The constructor for this class may generate
        some of these required arguments if initially missing.
        """
        self.dataset = dataset
        self.num_classes = len(classes)
        self.nodes = Node.get_nodes(path_graph, path_wnids, classes)
        self.rules = Rules(dataset, path_graph, path_wnids, classes)
        self.tree_supervision_weight = tree_supervision_weight
        self.criterion = criterion

    @staticmethod
    def assert_output_not_nbdt(outputs):
        """
        >>> x = torch.randn(1, 3, 224, 224)
        >>> TreeSupLoss.assert_output_not_nbdt(x)  # all good!
        >>> x._nbdt_output_flag = True
        >>> TreeSupLoss.assert_output_not_nbdt(x)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        >>> from nbdt.model import NBDT
        >>> import torchvision.models as models
        >>> model = models.resnet18()
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  # all good!
        >>> model = NBDT('CIFAR10', model, arch='ResNet18')
        >>> y = model(x)
        >>> TreeSupLoss.assert_output_not_nbdt(y)  #doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        AssertionError: ...
        """
        assert getattr(outputs, '_nbdt_output_flag', False) is False, (
            "Uh oh! Looks like you passed an NBDT model's output to an NBDT "
            "loss. NBDT losses are designed to take in the *original* model's "
            "outputs, as input. NBDT models are designed to only be used "
            "during validation and inference, not during training. Confused? "
            " Check out github.com/alvinwan/nbdt#convert-neural-networks-to-decision-trees"
            " for examples and instructions.")


class HardTreeSupLoss(TreeSupLoss):

    def forward(self, outputs, targets):
        """
        The supplementary losses are all uniformly down-weighted so that on
        average, each sample incurs half of its loss from standard cross entropy
        and half of its loss from all nodes.

        The code below is structured weirdly to minimize number of tensors
        constructed and moved from CPU to GPU or vice versa. In short,
        all outputs and targets for nodes with 2 children are gathered and
        moved onto GPU at once. Same with those with 3, with 4 etc. On CIFAR10,
        the max is 2. On CIFAR100, the max is 8.
        """
        # print("begin Hard Loss forward")
        self.assert_output_not_nbdt(outputs)

        loss = self.criterion(outputs, targets)
        maxk = max((1,2))  # 取top1准确率，若取top1和top5准确率改为max((1,5))
        m = nn.Softmax(dim=1)
        value, pred = m(outputs).topk(maxk, 1, True, True)

        print(m(outputs))
        # print(outputs.T)
        print(value)
        print(value.var())
        # print(torch.max(outputs.T,0)[0])
        # print(torch.max(outputs.T,0)[1])

        # print("loss")
        # print(loss)
        num_losses = outputs.size(0) * len(self.nodes) / 2.
        # print("outputs")
        # print(outputs)
        # for node in self.nodes:
        #     print("node.synset")
        #     print(node.synset)
        outputs_subs = defaultdict(lambda: [])
        targets_subs = defaultdict(lambda: [])
        targets_ints = [int(target) for target in targets.cpu().long()]
        for node in self.nodes:
            _, outputs_sub, targets_sub = \
                HardEmbeddedDecisionRules.get_node_logits_filtered(
                    node, outputs, targets_ints)
            # print("node.synset")
            # print(node.synset)
            # print("outputs_sub")
            # print(outputs_sub)
            # print("targets_sub")
            # print(targets_sub)
            # print("targets_ints")
            # print(targets_ints)
            key = node.num_classes
            # print("key")
            # print(key)
            assert outputs_sub.size(0) == len(targets_sub)
            outputs_subs[key].append(outputs_sub)
            targets_subs[key].extend(targets_sub)
        # print("outputs_subs")
        # print(outputs_subs)
        # print("targets_subs")
        # print(targets_subs)

        for key in outputs_subs:
            outputs_sub = torch.cat(outputs_subs[key], dim=0)
            targets_sub = torch.Tensor(targets_subs[key]).long().to(outputs_sub.device)
            # print("key")
            # print(key)
            # print("outputs_sub")
            # print(outputs_sub)
            # print("targets_sub")
            # print(targets_sub)
            if not outputs_sub.size(0):
                continue
            fraction = outputs_sub.size(0) / float(num_losses) \
                * self.tree_supervision_weight
            loss += self.criterion(outputs_sub, targets_sub) * fraction
        # print("end Hard Loss forward")
            # output_nodes = torch.chunk(outputs_sub,outputs_sub.size(0),0)
            # print(loss)

            # for i in range(outputs_sub.size(0)):
            #     target_node = torch.Tensor(targets_subs[key][i:i+1]).long().to(outputs_sub.device)
            #     output_node = output_nodes[i]
            #     loss_tmp = self.criterion(output_node, target_node) *(1-i/float(outputs_sub.size(0)))
            #     loss = torch.max(loss_tmp,loss)
                # print(loss_tmp)

        return loss


class SoftTreeSupLoss(TreeSupLoss):

    def __init__(self, *args, Rules=None, **kwargs):
        super().__init__(*args, Rules=SoftEmbeddedDecisionRules, **kwargs)

    def forward(self, outputs, targets):
        self.assert_output_not_nbdt(outputs)
        # print("outputs")
        # print(outputs)

        loss = self.criterion(outputs, targets)
        bayesian_outputs = self.rules(outputs)
        # print("bayesian_outputs")
        # print(bayesian_outputs)
        loss_tmp= self.criterion(bayesian_outputs, targets) * self.tree_supervision_weight
        # print(loss)
        # print(loss_tmp)
        return loss
