import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary

# base libraries
import argparse
import os
import setproctitle
import shutil

# internals
from src import *

DATA_DIR = os.path.join(os.path.expanduser('~'), 'data/kaggle_protein')

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)
    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('-l', '--load')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 of Adam. Default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 of Adam. Default=0.999')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--valBatchSize', type=int, default=10, help='validating batch size')
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--lambda1', type=float, default=0.5, help='Lambda1 of L1. Default=0.5')
    parser.add_argument('--lambda2', type=float, default=0.01, help='Lambda2 of Lambda2. Default=0.01')

    parser.add_argument('--showGPU', type=str, default="0,1,2,3", help='GPUs for CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--useGPU', type=str, default="0,1,2,3", help='GPUs to use.')

    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))

    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--split', action='store_true', help="If create val set.")
    parser.add_argument('--subsample', action='store_true', help="If true train on subsample of images to test locally.")

    args = parser.parse_args()
    print(args)

    return args

def main():
    args.save = 'work/%s/%s' % (args.network_name, args.dataset_name)
    setproctitle.setproctitle(args.save)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    ################################ device ##############################

    print("\nCreating architecture...")
    torch.cuda.manual_seed_all(args.seed)

    if args.load:
        print("Loading network: {}".format(args.load))
        net = torch.load(args.load)
    else:
        net = get_network(args.pretrained)

    device_ids = list(map(int, args.useGPU.split(',')))
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=device_ids)

    summary(net, input_size=(4, 512, 512))

    ################################ data ##############################

    print("\nLoading data...")
    dataset = get_dataset(DATA_DIR, mode='train')
    if args.subsample:
        n_subsample = 100
        new_dataset = []
        for i in range(n_subsample):
            new_dataset.append(dataset[i])
        dataset = new_dataset

    if args.split:
        trainset, valset = get_train_val_split(dataset)
        trainLoader = DataLoader(dataset=trainset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=True)
        valLoader = DataLoader(dataset=valset, batch_size=args.valBatchSize, num_workers=4, pin_memory=True, shuffle=False)
    else:
        trainLoader = DataLoader(dataset=dataset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=False)

    ################################ criterion and optimizer ##############################

    print("\nCreating criterion and optimizer...")
    # criterion = get_loss_function()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, weight_decay=1e-4, lr=args.lr)
    elif args.opt == 'adam':
        if args.lr == 0.001:
            optimizer = torch.optim.Adam(net.parameters(), betas=(args.beta1, args.beta2))
        else:
            optimizer = torch.optim.Adam(net.parameters(), betas=(args.beta1, args.beta2), lr=args.lr)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
    else:
        raise ModuleNotFoundError('optimiser not found')

    ################################ train ##############################

    print("\nTraining...")
    for epoch in range(1, args.nEpochs):
        train(args, epoch, net, trainLoader, criterion, optimizer)
        if args.split:
            val(args, net, valLoader, criterion)

        if epoch % 10 == 0:
            torch.save(net, os.path.join(args.save, '%d.pth' % epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
        print("\n")

def train(args, epoch, net, trainLoader, criterion, optimizer):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (imagename, inputs, labels) in enumerate(trainLoader):
        # get the inputs
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)       # outputs.shape labels.shape 16*28
        loss = criterion(outputs, labels)
        # print(loss)

        ######################### mean_loss ######################
        # mean_loss = loss.mean()
        ######################### sum_loss #######################
        # sum_loss = loss.sum()

        # loss = loss.mean() + args.lambda2 * l2_penalty(outputs)
        loss = loss + args.lambda2 * l2_penalty(outputs)
        loss.backward()

        # sum_loss += args.lambda2 * l2_penalty(outputs)
        # sum_loss.backward()
        optimizer.step()

        nProcessed += len(inputs)

        pred = torch.sigmoid(outputs).data.gt(0.5)
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
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader), loss.item()))
        for param_lr in optimizer.param_groups:
            print('learning_rate:' + str(param_lr['lr']))
        # if batch_idx % args.batchSize == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader), loss.item()))
        #     # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}\tPrec: {:.6f}\tRec: {:.6f}\tTP: {:^6}\tFP: {:^6}\tFN: {:^6}\tTN: {:^6}'.format(
        #     #     epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
        #     #     loss.item(), acc, prec, rec, tp, fp, fn, tn))
        #     for param_lr in optimizer.param_groups:
        #         print('learning_rate:' + str(param_lr['lr']))

def val(args, net, valLoader, criterion):
    net.eval()
    val_loss = 0
    acc = prec = rec = 0
    incorrect = 0
    for batch_idx, data in enumerate(valLoader):
        inputs, labels = data['image'], data['labels']

        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = net(inputs)
        val_loss += criterion(outputs, labels)

        pred = torch.sigmoid(outputs).data.gt(0.5)
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

    val_loss /= len(valLoader)
    acc /= len(valLoader)
    prec /= len(valLoader)
    rec /= len(valLoader)

    print('\nValidation set: Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n'.format(val_loss, acc, prec, rec))


if __name__ == '__main__':
    args = get_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.showGPU
    main()