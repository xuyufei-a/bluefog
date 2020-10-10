# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import tensorboardX
from tqdm import tqdm

from bluefog.common import topology_util
import bluefog.torch as bf

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--dist-optimizer', type=str, default='neighbor_allreduce',
                    help='The type of distributed optimizer. Supporting options are ' +
                    '[neighbor_allreduce, allreduce, win_put, horovod]')
parser.add_argument("--disable-dynamic-topology", action="store_true",
                    default=False, help=("Disable each iteration to transmit one neighbor " +
                                         "per iteration dynamically."))
parser.add_argument('--virtual-topology', type=str, default="power2",
                    help='The underlying virtual topology. Supporting options are ' +
                    '[power2(Default), ring, mesh, star].')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

if args.dist_optimizer == 'horovod':
    print("importing horovod")
    import horovod.torch as bf

bf.init()
torch.manual_seed(args.seed)

if args.dist_optimizer != 'horovod':
    if args.virtual_topology == "power2":
        pass
    elif args.virtual_topology == "ring":
        bf.set_topology(topology_util.RingGraph(bf.size(), connect_style=0))
    elif args.virtual_topology == "mesh":
        bf.set_topology(topology_util.RingGraph(
            bf.size(), connect_style=0), is_weighted=True)
    elif args.virtual_topology == "star":
        bf.set_topology(topology_util.StarGraph(bf.size()))
    else:
        raise ValueError("Unknown args.virtual_topology, supporting options are " +
                         "[power2(Default), ring, mesh, star].")

if args.cuda:
    # Bluefog: pin GPU to local rank.
    torch.cuda.set_device(bf.local_rank() % torch.cuda.device_count())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Bluefog: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = bf.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                 name='resume_from_epoch').item()

# Bluefog: print logs on the first worker.
verbose = 1 if bf.rank() == 0 else 0

# Bluefog: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if bf.rank() == 0 else None


# Bluefog: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# Bluefog: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=bf.size()` and `rank=bf.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bf.size(), rank=bf.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bf.size(), rank=bf.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


# Set up standard ResNet-50 model.å
model = models.resnet50()

# For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
lr_scaler = args.batches_per_allreduce * bf.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Bluefog: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          lr_scaler),
                      momentum=args.momentum, weight_decay=args.wd)

# Bluefog: wrap optimizer with DistributedOptimizer.
# TODO(hhb) Pass  batches_per_allreduce to optimizer.
if args.dist_optimizer == 'win_put':
    optimizer = bf.DistributedBluefogOptimizer(optimizer, model=model)
elif args.dist_optimizer == 'neighbor_allreduce':
    optimizer = optimizer = bf.DistributedNeighborAllreduceOptimizer(
        optimizer, model=model)
elif args.dist_optimizer == 'allreduce':
    optimizer = optimizer = bf.DistributedAllreduceOptimizer(
        optimizer, model=model)
elif args.dist_optimizer == 'horovod':
    optimizer = optimizer = bf.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
else:
    raise ValueError('Unknown args.dist-optimizer type -- ' + args.dist_optimizer + '\n' +
                     'Please set the argument to be one of ' +
                     '[neighbor_allreduce, allreduce, win_put, horovod]')

# Restore from a previous checkpoint, if initial_epoch is specified.
# Bluefog: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and bf.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Bluefog: broadcast parameters & optimizer state.
bf.broadcast_parameters(model.state_dict(), root_rank=0)
bf.broadcast_optimizer_state(optimizer, root_rank=0)

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Bluefog: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / bf.size() * (epoch * (bf.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * bf.size() * args.batches_per_allreduce * lr_adj

if not args.disable_dynamic_topology and args.dist_optimizer != 'horovod':
    dynamic_neighbor_allreduce_gen = topology_util.GetDynamicSendRecvRanks(
        bf.load_topology(), bf.rank())

def dynamic_topology_update(epoch, batch_idx):
    if args.dist_optimizer == 'win_put':
        if epoch < 3:
            return
        num_out_neighbors = len(bf.out_neighbor_ranks())
        sent_neighbor = bf.out_neighbor_ranks()[batch_idx % num_out_neighbors]
        optimizer.dst_weights = {sent_neighbor: 1.0}
    elif args.dist_optimizer == 'neighbor_allreduce':
        send_neighbor, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
        optimizer.send_neighbors = [send_neighbor]
        optimizer.neighbor_weights = {r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
        optimizer.self_weight = 1 / (len(recv_neighbors) + 1)
        optimizer.enable_topo_check = False
    else:
        pass

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if bf.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Bluefog: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
    save_checkpoint(epoch)
