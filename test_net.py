import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nbdt import data, analysis, loss, models
import torchvision
import torchvision.transforms as transforms
from nbdt.model import SoftNBDT
from nbdt.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from IPython.display import display
import os
import argparse
import numpy as np

from nbdt.utils import (
    progress_bar, generate_fname, generate_kwargs, Colors, maybe_install_wordnet
)

maybe_install_wordnet()

datasets = ('CIFAR10', 'CIFAR100') + data.imagenet.names + data.custom.names

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='CIFAR10', choices=datasets)
parser.add_argument('--arch', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# extra general options for main script
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--name', default='',
                    help='Name of experiment. Used for checkpoint filename')
parser.add_argument('--pretrained', action='store_true',
                    help='Download pretrained model. Not all models support this.')
parser.add_argument('--eval', help='eval only', action='store_true')

# options specific to this project and its dataloaders
parser.add_argument('--loss', choices=loss.names, default='CrossEntropyLoss')
parser.add_argument('--analysis', choices=analysis.names, help='Run analysis after each epoch')
parser.add_argument('--input-size', type=int,
                    help='Set transform train and val. Samples are resized to '
                    'input-size + 32.')
parser.add_argument('--lr-decay-every', type=int, default=0)

data.custom.add_arguments(parser)
loss.add_arguments(parser)
analysis.add_arguments(parser)

args = parser.parse_args()

loss.set_default_values(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = getattr(data, args.dataset)

dataset_kwargs = generate_kwargs(args, dataset,
    name=f'Dataset {args.dataset}',
    keys=data.custom.keys,
    globals=globals())

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

Colors.cyan(f'Training with dataset {args.dataset} and {len(trainset.classes)} classes')

# Model
print('==> Building model..')
model = getattr(models, args.arch)
model_kwargs = {'num_classes': len(trainset.classes) }

if args.pretrained:
    print('==> Loading pretrained model..')
    net = model(pretrained=True, dataset=args.dataset, **model_kwargs)
else:
    print('==> Loading NBDT model..')
    net = ResNet18()
    net = SoftNBDT(pretrained=True, dataset='CIFAR10', arch='ResNet18', model=net)

#Analyzer
class_analysis = getattr(analysis, args.analysis or 'Noop')
analyzer_kwargs = generate_kwargs(args, class_analysis,
    name=f'Analyzer {args.analysis}',
    keys=analysis.keys,
    globals=globals())
analyzer = class_analysis(**analyzer_kwargs)

#Criterion
criterion = nn.CrossEntropyLoss()
class_criterion = getattr(loss, args.loss)
loss_kwargs = generate_kwargs(args, class_criterion,
    name=f'Loss {args.loss}',
    keys=loss.keys,
    globals=globals())
criterion = class_criterion(**loss_kwargs)

#Test
def test(epoch, analyzer):
  analyzer.start_test(epoch)

  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
          inputs, targets = inputs.to(device), targets.to(device)
          outputs = net(inputs)
          loss = criterion(outputs, targets)

          test_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()

          if device == 'cuda':
              predicted = predicted.cpu()
              targets = targets.cpu()

          predicted_nbdt = analyzer.update_batch(outputs, targets)
          print(predicted.eq(targets).sum())
          print(predicted_nbdt.eq(targets).sum())
          progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) '
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


  acc = 100.*correct/total
  print("Accuracy: {}, {}/{}".format(acc, correct, total))

  analyzer.end_test(epoch)

#Evaluation
analyzer.start_epoch(0)
test(0, analyzer)

#example
#python3 test_net.py --dataset=CIFAR10 --arch=ResNet18  --loss=SoftTreeSupLoss --pretrained/-eval --analysis=SoftEmbeddedDecisionRules
