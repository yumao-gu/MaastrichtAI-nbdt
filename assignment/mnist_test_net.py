import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from assignment import data, analysis, loss, models
from assignment.mnist_model_generate import Net
from assignment.log import Logger
import torchvision
import torchvision.transforms as transforms
from assignment.model import SoftNBDT
from assignment.models import ResNet18, wrn28_10_cifar10, wrn28_10_cifar100, wrn28_10  # use wrn28_10 for TinyImagenet200
from torchvision import transforms
from nbdt.utils import DATASET_TO_CLASSES, load_image_from_path, maybe_install_wordnet
from IPython.display import display
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

from assignment.utils import (
    progress_bar, generate_fname, generate_kwargs, Colors, maybe_install_wordnet
)

maybe_install_wordnet()

datasets = ('CIFAR10', 'CIFAR100','MNIST') + data.imagenet.names + data.custom.names

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=100, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='MNIST', choices=datasets)
parser.add_argument('--arch', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# extra general options for main script
parser.add_argument('--path-resume', default='',
                    help='Overrides checkpoint path generation')
parser.add_argument('--data-folder', default='',
                    help='the file folder makes')
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

###logging and saving images
folder = "./data/MNIST/"+args.data_folder
if not os.path.exists(folder):
  os.makedirs(folder)
log_file_name = folder +"/log.txt"
logger = Logger(log_file_name, log_level=logging.INFO, logger_name="decision").get_log()

# Data
print('==> Preparing data..')
transform_=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

dataset = getattr(data, args.dataset)

dataset_kwargs = generate_kwargs(args, dataset,
    name=f'Dataset {args.dataset}',
    keys=data.custom.keys,
    globals=globals())

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset,batch_size=1)

Colors.cyan(f'Testing with dataset {args.dataset} and {len(testset.classes)} classes')

# Model
# TODO(alvin): fix checkpoint structure so that this isn't neededd
def load_state_dict(net, state_dict):
    try:
        net.load_state_dict(state_dict)
    except RuntimeError as e:
        if 'Missing key(s) in state_dict:' in str(e):
            net.load_state_dict({
                key.replace('module.', '', 1): value
                for key, value in state_dict.items()})

if args.pretrained:
    net_old = Net()
    print('==> Loading pretrained model..')
    checkpoint_old = torch.load('./mnist_cnn_200.pth', map_location=torch.device(device))
    if 'net' in checkpoint_old:
        load_state_dict(net_old,checkpoint_old['net'])
    else:
        load_state_dict(net_old, checkpoint_old)

net = Net()
print('==> Loading NBDT model..')
checkpoint_path = args.path_resume
checkpoint= torch.load(checkpoint_path, map_location=torch.device(device))
if 'net' in checkpoint:
    load_state_dict(net, checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    Colors.cyan(f'==> Checkpoint found for epoch {start_epoch} with accuracy '
          f'{best_acc} at {checkpoint_path}')
else:
    load_state_dict(net,checkpoint)

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

#inverse_transform
def inverse_transform(input,mean,std):
    input[0]=input[0]*std[0]+mean[0]
    # input[1]=input[1]*std[1]+mean[1]
    # input[2]=input[2].mul(std[2])+mean[2]
    img = input.mul(255).byte()
    # img = img.numpy().transpose((1, 2, 0))
    return img

#Test
def test(epoch, analyzer):
  analyzer.start_test(epoch)
  net.eval()
  net_old.eval()
  test_loss = 0
  correct, correct_old, correct_nbdt = 0, 0, 0
  total = 0
  wrong_number = 0
  with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
          inputs, targets = inputs.to(device), targets.to(device)
          outputs = net(inputs)
          outputs_old = net_old(inputs)
          loss = criterion(outputs, targets)

          test_loss += loss.item()
          _, predicted = outputs.max(1)
          _, predicted_old = outputs_old.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()
          correct_old += predicted_old.eq(targets).sum().item()

          if device == 'cuda':
              predicted = predicted.cpu()
              targets = targets.cpu()

          predicted_nbdt,decisions_nbdt = analyzer.update_batch(outputs, targets)
          correct_nbdt += predicted_nbdt.eq(targets).sum().item()
          if not predicted_nbdt.eq(targets) and not predicted_old.eq(targets):
              wrong_number += 1
              print("something good happens")
              img = inputs[0]
              # img = inverse_transform(inputs[0],(0.1307,), (0.3081,))
              image = img.numpy()[0]
              imagefile = folder+'/'+str(predicted_old.numpy()[0])\
                +'-'+str(predicted_nbdt.numpy()[0])+'-'+str(targets.numpy()[0])+".png"
              plt.imsave(imagefile,img[0])
              message = '\n\n'+str(outputs)+' '+str(predicted_old.numpy()[0])\
                +'-'+str(predicted_nbdt.numpy()[0])+'-'+str(targets.numpy()[0])+'\n'\
                +str(decisions_nbdt)+'\n'+str(outputs_old)+'\n'
              logger.info(message)

          progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f | Acc_old: %.3f | Acc_nbdt: %.3f'
              % (test_loss/(batch_idx+1), 100.*correct/total, 100.*correct_old/total, 100.*correct_nbdt/total))


  acc = 100.*correct/total
  acc_old = 100.*correct_old/total
  acc_nbdt = 100.*correct_nbdt/total
  print("Accuracy: {},   {},   {}".format(acc, acc_old, acc_nbdt))
  print("Wrong number:{}".format(wrong_number))

  analyzer.end_test(epoch)

#Evaluation
analyzer.start_epoch(0)
test(0, analyzer)

#example
#python3 mnist_test_net.py --loss=SoftTreeSupLoss --pretrained --path-resume ./checkpoint/ckpt-ave-l1-MNIST-SoftTreeSupLoss.pth --path-graph ./graph-ave-l1-MNIST.json --analysis=SoftEmbeddedDecisionRules --data-folder ave-l1-Soft_predict-exp
