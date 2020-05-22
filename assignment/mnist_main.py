import torch
from  assignment.mnist_model_generate import Net
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from nbdt import data, analysis, models
from assignment import loss,models,data, analysis
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from assignment.utils import (
    progress_bar, generate_fname, generate_kwargs, Colors, maybe_install_wordnet
)

maybe_install_wordnet()

datasets = ('CIFAR10', 'CIFAR100','MNIST') + data.imagenet.names + data.custom.names

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch-size', default=512, type=int,
                    help='Batch size used for training')
parser.add_argument('--epochs', '-e', default=20, type=int,
                    help='By default, lr schedule is scaled accordingly')
parser.add_argument('--dataset', default='MNIST', choices=datasets)
parser.add_argument('--arch', default='ResNet18', choices=list(models.get_model_choices()))
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# extra general options for main script
parser.add_argument('--path-resume', default='./mnist_cnn.pth',
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
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

dataset = getattr(data, args.dataset)

# if args.dataset in ('TinyImagenet200', 'Imagenet1000'):
#     default_input_size = 64 if args.dataset == 'TinyImagenet200' else 224
#     input_size = args.input_size or default_input_size
#     transform_train = dataset.transform_train(input_size)
#     transform_test = dataset.transform_val(input_size)
# elif args.input_size is not None and args.input_size > 32:
#     transform_train = transforms.Compose([
#         transforms.Resize(args.input_size + 32),
#         transforms.RandomCrop(args.input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     transform_test = transforms.Compose([
#         transforms.Resize(args.input_size + 32),
#         transforms.CenterCrop(args.input_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
dataset_kwargs = generate_kwargs(args, dataset,
    name=f'Dataset {args.dataset}',
    keys=data.custom.keys,
    globals=globals())

trainset = dataset(**dataset_kwargs, root='./data', train=True, download=True, transform=transform_train)
testset = dataset(**dataset_kwargs, root='./data', train=False, download=True, transform=transform_test)

assert trainset.classes == testset.classes, (trainset.classes, testset.classes)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size, shuffle=True, num_workers=2)
# trainloader = torch.utils.data.DataLoader(trainset)
# testloader = torch.utils.data.DataLoader(testset)
testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle=True, num_workers=2)


Colors.cyan(f'Training with dataset {args.dataset} and {len(trainset.classes)} classes')


# Model
print('==> Building model..')
# model = getattr(models, args.arch)
# model_kwargs = {'num_classes': len(trainset.classes) }
#
# if args.pretrained:
#     print('==> Loading pretrained model..')
#     try:
#         net = model(pretrained=True, dataset=args.dataset, **model_kwargs)
#     except TypeError as e:  # likely because `dataset` not allowed arg
#         print(e)
#
#         try:
#             net = model(pretrained=True, **model_kwargs)
#         except Exception as e:
#             Colors.red(f'Fatal error: {e}')
#             exit()
# else:
#     net = model(**model_kwargs)
#
# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
#
checkpoint_fname = generate_fname(**vars(args))
checkpoint_path = './checkpoint/{}.pth'.format(checkpoint_fname)
print(f'==> Checkpoints will be saved to: {checkpoint_path}')
#
#
# TODO(alvin): fix checkpoint structure so that this isn't neededd
# def load_state_dict(state_dict):
#     try:
#         net.load_state_dict(state_dict)
#     except RuntimeError as e:
#         if 'Missing key(s) in state_dict:' in str(e):
#             net.load_state_dict({
#                 key.replace('module.', '', 1): value
#                 for key, value in state_dict.items()
#             })

net = Net()
resume_path = args.path_resume # or checkpoint_path
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if not os.path.exists(resume_path):
        print('==> No checkpoint found. Skipping...')
    else:
        checkpoint = torch.load(resume_path, map_location=torch.device(device))
        if 'net' in checkpoint:
            net = net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            Colors.cyan(f'==> Checkpoint found for epoch {start_epoch} with accuracy '
                  f'{best_acc} at {resume_path}')
        else:
            net.load_state_dict(checkpoint)
            Colors.cyan(f'==> Checkpoint found at {resume_path}')
            # print(net.conv1.weight)


criterion = nn.CrossEntropyLoss()
class_criterion = getattr(loss, args.loss)
loss_kwargs = generate_kwargs(args, class_criterion,
    name=f'Loss {args.loss}',
    keys=loss.keys,
    globals=globals())
criterion = class_criterion(**loss_kwargs)
#
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#
# def adjust_learning_rate(epoch, lr):
#     if args.lr_decay_every:
#         steps = epoch // args.lr_decay_every
#         return lr / (10 ** steps)
#     if epoch <= 150 / 350. * args.epochs:  # 32k iterations
#         return lr
#     elif epoch <= 250 / 350. * args.epochs:  # 48k iterations
#         return lr/10
#     else:
#         return lr/100
#
# Training
def train(epoch, analyzer):
    analyzer.start_train(epoch)
    # lr = adjust_learning_rate(epoch, args.lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        stat,_ = analyzer.update_batch(outputs, targets)
        # extra = f'| {stat}' if stat else ''
        extra=''

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) %s'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, extra))

    analyzer.end_train(epoch)

def test(epoch, analyzer, checkpoint=True):
    analyzer.start_test(epoch)
    global best_acc
    net.eval()
    test_loss = 0
    correct, correct_nbdt = 0, 0
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

            predicted_nbdt,decisions_nbdt = analyzer.update_batch(outputs, targets)
            correct_nbdt += predicted_nbdt.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct_nbdt/total, correct_nbdt, total))
    acc = 100.*correct_nbdt/total
    print("Accuracy: {}, {}/{}".format(acc, correct_nbdt, total))
    if acc > best_acc: # and checkpoint:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        print(f'Saving to {checkpoint_fname} ({acc})..')
        torch.save(state, f'./checkpoint/{checkpoint_fname}.pth')
        best_acc = acc
    analyzer.end_test(epoch)

class_analysis = getattr(analysis, args.analysis or 'Noop')
analyzer_kwargs = generate_kwargs(args, class_analysis,
    name=f'Analyzer {args.analysis}',
    keys=analysis.keys,
    globals=globals())
analyzer = class_analysis(**analyzer_kwargs)

#
# if args.eval:
#     if not args.resume and not args.pretrained:
#         Colors.red(' * Warning: Model is not loaded from checkpoint. '
#         'Use --resume or --pretrained (if supported)')
#
#     analyzer.start_epoch(0)
#     test(0, analyzer, checkpoint=False)
#     exit()
#
for epoch in range(start_epoch, args.epochs):
    analyzer.start_epoch(epoch)
    train(epoch, analyzer)
    test(epoch, analyzer)
    analyzer.end_epoch(epoch)
#
# if args.epochs == 0:
#     analyzer.start_epoch(0)
#     test(0, analyzer)
#     analyzer.end_epoch(0)
# print(f'Best accuracy: {best_acc} // Checkpoint name: {checkpoint_fname}')

#python3 mnist_main.py --resume --path_graph=./graph-MNIST.json --loss=SoftTreeSupLoss
