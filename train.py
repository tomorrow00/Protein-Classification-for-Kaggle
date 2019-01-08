import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim import lr_scheduler

# base libraries
import argparse
import os
import setproctitle
import shutil
import warnings
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# internals
from src import *
from config import config

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.expanduser('~'), 'data/kaggle/protein')

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)

    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-s', '--split', action='store_true', default=True, help="If create val set.")
    parser.add_argument('-sub', '--subsample', action='store_true',
                        help="If true train on subsample of images to test locally.")

    parser.add_argument('--architecture', type=str, default=config.architecture,
                        help='Choose architecture from custom, resnet, bcnn, densenet, inception, squeezenet')

    parser.add_argument('--lr', type=float, default=config.lr, help='Learning Rate. Default=0.01')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 of Adam. Default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 of Adam. Default=0.999')
    parser.add_argument('--batchSize', type=int, default=config.batchSize, help='training batch size')
    parser.add_argument('--valBatchSize', type=int, default=10, help='validating batch size')
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--lambda1', type=float, default=0.5, help='Lambda1 of L1. Default=0.5')
    parser.add_argument('--lambda2', type=float, default=0.01, help='Lambda2 of Lambda2. Default=0.01')

    parser.add_argument('--showGPU', type=str, default=config.showGPU, help='GPUs for CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--useGPU', type=str, default=config.useGPU, help='GPUs to use.')

    parser.add_argument('--opt', type=str, default=config.opt, choices=('sgd', 'adam', 'rmsprop'))

    parser.add_argument('--seed', type=int, default=2050)

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

    net = get_network(args.architecture, args.pretrained)
    if args.load:
        # loadfile = sorted(os.listdir(os.path.join('work/%s/%s' % (args.network_name, 'pretrained'))), key=lambda x:int(x.split(".")[0]), reverse=True)[0]
        loadfile = 'best_loss.pth'
        # loadfile = 'best_f1.pth'
        loadfile = os.path.join('work/%s/%s' % (args.network_name, '1'), loadfile)
        print("Loading network: {}".format(loadfile))
        # m = torch.load(loadfile)
        # print(m)
        # module.conv1_7x7_s2.weight
        # net.load_state_dict(torch.load(loadfile))
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(loadfile).items()})


    net = net.cuda()
    device_ids = list(map(int, args.useGPU.split(',')))
    net = torch.nn.DataParallel(net, device_ids=device_ids)

    # summary(net, input_size=(3, 512, 512))
    # summary(net, input_size=(4, 512, 512))
    # print(net)

    ################################ data ##############################

    print("\nLoading data...")
    dataset = get_dataset(DATA_DIR, mode='train', split=args.split, subsample=args.subsample, folds=config.folds, foldnum=config.foldnum-1)

    if args.split:
        trainset, valset = dataset
        print("Trainset:", len(trainset), "\tValset:", len(valset))
        trainLoader = DataLoader(dataset=trainset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=True)
        valLoader = DataLoader(dataset=valset, batch_size=args.valBatchSize, num_workers=4, pin_memory=True, shuffle=False)
    else:
        trainLoader = DataLoader(dataset=dataset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=False)

    ################################ criterion and optimizer ##############################

    print("\nCreating criterion and optimizer...")
    print("criterion with BCEWithLogitsLoss, optimizer with " + args.opt)

    # criterion = get_loss_function()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), betas=(args.beta1, args.beta2), lr=args.lr)
        # optimizer = torch.optim.Adam(net.parameters(), betas=(args.beta1, args.beta2))
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
    else:
        raise ModuleNotFoundError('optimizer not found')

    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    ################################ train ##############################

    print("\nTraining...")
    best_loss = 1000000
    best_f1 = best_precision = best_recall = 0
    for epoch in range(1, args.nEpochs):
        # params = list(net.parameters())
        # print(params[0][0][0][0][0])
        # print(params[-1][0])

        scheduler.step(epoch)

        train(args, epoch, net, trainLoader, criterion, optimizer)
        if args.split:
            loss, f1, precision, recall = val(net, valLoader, criterion)
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), os.path.join(args.save, 'best_loss.pth' % loss))
            if f1 > best_f1:
                best_f1 = f1
                torch.save(net.state_dict(), os.path.join(args.save, 'best_f1.pth' % f1))
            # if precision > best_precision:
            #     best_precision = precision
            #     torch.save(net, os.path.join(args.save, 'best_precision_%d.pth' % precision))
            # if recall > best_recall:
            #     best_recall = recall
            #     torch.save(net, os.path.join(args.save, 'best_recall_%d.pth' % recall))

        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(args.save, '%d.pth' % epoch))
            # torch.save(net, os.path.join(args.save, '%d.pth' % epoch))

        print("\n")

def train(args, epoch, net, trainLoader, criterion, optimizer):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (imagename, inputs, labels) in enumerate(trainLoader):
        if len(imagename) == 1:
            continue

        # get the inputs
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # forward + backward + optimizer
        outputs = net(inputs)       # outputs.shape labels.shape batchSize*28

        loss = criterion(outputs, labels)
        # loss = loss + args.lambda2 * l2_penalty(outputs)

        f1 = f1_score(labels.cpu(), outputs.sigmoid().cpu() > 0.15, average='macro')
        precision = precision_score(labels.cpu(), outputs.sigmoid().cpu() > 0.15, average='macro')
        recall = recall_score(labels.cpu(), outputs.sigmoid().cpu() > 0.15, average='macro')

        optimizer.zero_grad()       # zero the parameter gradients
        loss.backward()
        optimizer.step()

        nProcessed += len(inputs)

        print('Train Epoch: {} [{:^5}/{} ({:.0f}%)]\tLoss: {:.6f}\tf1_score: {:.6f}\tPrecision: {:.6f}\tRecall: {:.6f}'.format(
            epoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), f1, precision, recall))

        for param_lr in optimizer.param_groups:
            print('learning_rate:' + str(param_lr['lr']))

def val(net, valLoader, criterion):
    net.eval()

    f1 = precision = recall = val_loss = 0

    with torch.no_grad():
        for batch_idx, (imagename, inputs, labels) in enumerate(valLoader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = net(inputs)
            val_loss += criterion(outputs, labels)

            f1 += f1_score(labels, outputs.sigmoid().cpu() > 0.15, average='macro')
            precision += precision_score(labels, outputs.sigmoid().cpu() > 0.15, average='macro')
            recall += recall_score(labels, outputs.sigmoid().cpu() > 0.15, average='macro')

    val_loss /= len(valLoader)
    f1 /= len(valLoader)
    precision /= len(valLoader)
    recall /= len(valLoader)

    print('\nValidation set: Loss: {:.4f}\tf1_score: {:.4f}\tPrecision: {:.4f}\tRecall: {:.4f}\n'.format(val_loss, f1, precision, recall))

    return val_loss, f1, precision, recall

if __name__ == '__main__':
    print("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-----------------------------------------------'))
    args = get_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.showGPU
    main()

