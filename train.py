import torch
from torchvision import models
from torchsummary import summary

# base libraries
import argparse
import os
import setproctitle
import shutil

# internals
from src import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

BASE_DIR = '.'
DATA_DIR = '/home/wcc/data/kaggle_protein'
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_SPLIT = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)
    parser.add_argument('-m', '--multilabel', type=bool, default=False)
    parser.add_argument('-p', '--pretrained', type=bool, default=False)
    parser.add_argument('-dp', '--data-parallel', type=bool, default=True)
    parser.add_argument('-l', '--load')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 of Adam. Default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 of Adam. Default=0.999')
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size') # 64
    parser.add_argument('--valBatchSize', type=int, default=10, help='validating batch size')
    parser.add_argument('--nEpochs', type=int, default=2) # 300
    parser.add_argument('--sEpoch', type=int, default=1)
    parser.add_argument('--lambda1', type=float, default=0.5, help='Lambda1 of L1. Default=0.5')
    parser.add_argument('--lambda2', type=float, default=0.01, help='Lambda2 of Lambda2. Default=0.01')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--nGPU', type=int, default=0)
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--crit', type=str, default='bce', choices=('bce', 'f1'))
    parser.add_argument('--subsample', type=bool, default=False, help="If true train on subsample of images to test locally.")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda and args.nGPU == 0:
        nGPU = 1
    else:
        nGPU = args.nGPU

    args.save = args.save or 'work/%s/%s' % (args.network_name, args.dataset_name)
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    kwargs = {'num_workers': 4 * nGPU, 'pin_memory': True} if args.cuda and nGPU > 0 else {'num_workers': 4}

    dataset = get_dataset(TRAIN_IMAGE_DIR)

    trainLoader, valLoader = get_train_val_split(dataset, trainBS=args.batchSize, valBS=args.valBatchSize, val_split=VALIDATION_SPLIT, subsample=args.subsample, **kwargs)

    if args.load:
        print("Loading network: {}".format(args.load))
        net = torch.load(args.load)
    else:
        net = get_network(args.pretrained)

    if args.data_parallel:
        device_ids = [0, 1, 2, 3]
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    print('Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    if args.cuda:
        net = net.cuda()
    summary(net, input_size=(4, 512, 512))

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, weight_decay=1e-4, lr=args.lr)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), betas=(args.beta1, args.beta2), lr=args.lr)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
    else:
        raise ModuleNotFoundError('optimiser not found')

    criterion = get_loss_function(args.crit)

    # trainF = open(os.path.join(args.save, 'train.csv'), 'a')
    # valF = open(os.path.join(args.save, 'validation.csv'), 'a')

    for epoch in range(args.sEpoch, args.nEpochs + args.sEpoch):
        # adjust_opt(args.opt, optimizer, epoch)
        # train(args, epoch, net, trainLoader, criterion, optimizer, trainF)
        # val(args, epoch, net, valLoader, criterion, valF)
        train(args, epoch, net, trainLoader, criterion, optimizer, 0)
        # val(args, epoch, net, valLoader, criterion, 0)
        if epoch % 50 == 0:
            torch.save(net, os.path.join(args.save, '%d.pth' % epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
        print("\n")

    # trainF.close()
    # valF.close()

def train(args, epoch, net, trainLoader, criterion, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, data in enumerate(trainLoader):
        inputs, labels = data['image'], data['labels']

        # get the inputs
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) + args.lambda2 * l2_penalty(outputs)
        loss.backward()
        optimizer.step()
        nProcessed += len(inputs)

        if args.multilabel:
            pred = outputs.data.gt(0.5)
            tp = (pred + labels.data.byte()).eq(2).sum().float()
            fp = (pred - labels.data.byte()).eq(1).sum().float()
            fn = (pred - labels.data.byte()).eq(255).sum().float()
            tn = (pred + labels.data.byte()).eq(0).sum().float()
            acc = (tp + tn) / (tp + tn + fp + fn)
            try:
                prec = tp / (tp + fp)
            except ZeroDivisionError:
                prec = 0.0
            try:
                rec = tp / (tp + fn)
            except ZeroDivisionError:
                rec = 0.0
            if batch_idx % 10 == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                #     epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                #     loss.item(), acc))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}\tPrec: {:.6f}\tRec: {:.6f}\tTP: {:^6}\tFP: {:^6}\tFN: {:^6}\tTN: {:^6}'.format(
                    epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                    loss.item(), acc, prec, rec, tp, fp, fn, tn))
                for param_lr in optimizer.param_groups:
                    print('learning_rate:' + str(param_lr['lr']))
            # trainF.write('{},{},{},{},{}\n'.format(epoch, loss.item(), acc, prec, rec))
        else:
            pred = outputs.data.max(1)[1]
            incorrect = pred.ne(labels.data).sum()
            err = 100.*incorrect/len(data)
            partialEpoch = epoch + batch_idx / len(trainLoader) - 1
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                loss.data[0], err))
        #     trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        # trainF.flush()

def val(args, epoch, net, valLoader, criterion, valF):
    net.eval()
    val_loss = 0
    acc = prec = rec = 0
    incorrect = 0
    for batch_idx, data in enumerate(valLoader):
        inputs, labels = data['image'], data['labels']

        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs)
        val_loss += criterion(outputs, labels)

        if args.multilabel:
            pred = outputs.data.gt(0.5)
            tp = (pred + labels.data.byte()).eq(2).sum().float()
            fp = (pred - labels.data.byte()).eq(1).sum().float()
            fn = (pred - labels.data.byte()).eq(255).sum().float()
            tn = (pred + labels.data.byte()).eq(0).sum().float()
            acc += (tp + tn) / (tp + tn + fp + fn)
            try:
                prec += tp / (tp + fp)
            except ZeroDivisionError:
                prec += 0.0
            try:
                rec += tp / (tp + fn)
            except ZeroDivisionError:
                rec += 0.0
        else:
            pred = outputs.data.max(1)[1] # get the index of the max log-probability
            incorrect += pred.ne(labels.data).sum()

    val_loss /= len(valLoader)
    acc /= len(valLoader)
    prec /= len(valLoader)
    rec /= len(valLoader)

    if args.multilabel:
        print('\nValidation set: Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n'.format(
            val_loss, acc, prec, rec))
        # valF.write('{},{},{},{},{}\n'.format(epoch, val_loss, acc, prec, rec))
    else:
        nTotal = len(valLoader.dataset)
        err = 100. * incorrect / nTotal
        print('\nValidation set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        val_loss, incorrect, nTotal, err))
        # valF.write('{},{},{}\n'.format(epoch, val_loss, err))

    # valF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())

if __name__ == '__main__':
    main()
