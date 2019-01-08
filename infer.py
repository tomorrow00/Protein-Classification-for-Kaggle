import torch
from torchsummary import summary
from torch.utils.data import DataLoader

# base libraries
import os
import argparse
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# internals
from src import *
from config import config

torch.backends.cudnn.benchmark = True

NUM_ALL = 98699
NUM1 = 31072
NUM2 = 67627
PROTEIN_NUM2 = [12885, 1254, 3621, 1561, 1858, 2513, 1008,
                2822, 53, 45, 28, 1093, 688, 537,
                1066, 21, 530, 210, 902, 1482, 172,
                3777, 802, 2965, 322, 8228, 328, 11]

PROTEIN_NUM1 = [24518, 1356, 5288, 1561, 2705, 2323, 2076,
                4518, 8361, 8324, 8324, 693, 1154, 557,
                1269, 20, 474, 72, 559, 1499, 8397,
                6969, 109, 5755, 178, 22938, 295, 46]

DATA_DIR = os.path.join(os.path.expanduser('~'), 'data/kaggle/protein')

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)

    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('--architecture', type=str, default=config.architecture,
                        help='Choose architecture from custom, resnet, bcnn, densenet, inception, squeezenet')

    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')

    parser.add_argument('--showGPU', type=str, default="1", help='GPUs for CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--useGPU', type=str, default="0", help='GPUs to use.')

    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    print(args)

    return args

def main():
    args.save = 'work/%s/%s' % (args.network_name, args.dataset_name)

    ################################ device ##############################

    print("\nCreating architecture...")
    torch.cuda.manual_seed_all(args.seed)

    # args.load = sorted(os.listdir(args.save), key=lambda x:int(x.split(".")[0]), reverse=True)[0]
    args.load = "best_loss.pth"
    # args.load = "best_f1.pth"
    loadforcsv = args.load.strip('.pth')
    args.load = os.path.join(args.save, args.load)
    print("Loading network: {}".format(args.load))

    net = get_network(args.architecture, args.pretrained)
    # net = torch.load(args.load)
    # net.load_state_dict(torch.load(args.load))
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load).items()})
    net = net.cuda()

    # device_ids = list(map(int, args.useGPU.split(',')))
    # net = torch.nn.DataParallel(net, device_ids=device_ids)

    # summary(net, input_size=(4, 512, 512))
    # summary(net, input_size=(3, 512, 512))

    ################################ data ##############################

    print("\nLoading data...")
    dataset = get_dataset(DATA_DIR, mode='test', split=False, subsample=False, folds=config.folds, foldnum=config.foldnum-1)
    testLoader = DataLoader(dataset=dataset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=False)

    csv_file = os.path.join('result', args.network_name + '_' + loadforcsv + '_' + str(datetime.datetime.now().strftime("%m-%d %H:%M")) + '.csv')
    print(csv_file)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    ################################ test ##############################

    print("\nTesting...")
    test(net, testLoader, csv_file)

def test(net, testLoader, csv_file):
    net.eval()
    writein = {}
    writein["Id"] = []
    writein["Predicted"] = []

    for batch_idx, (imagename, inputs) in enumerate(testLoader):
        inputs = inputs.cuda()
        outputs = net(inputs)

        print(batch_idx)
        print(imagename[0])

        # pred = torch.sigmoid(outputs).data.gt(0.4)
        pred = outputs.sigmoid().cpu().data.numpy()[0]

        prediction = []
        # q = []
        for i, p in enumerate(pred):
            # print('%.2f, %.2f, 0.15' % (p, float((PROTEIN_NUM1[i] + PROTEIN_NUM2[i]) / NUM_ALL)))
            # prediction.append(1 if p > float((PROTEIN_NUM1[i] + PROTEIN_NUM2[i]) / NUM_ALL) else 0)
            # q.append(1 if p > 0.15 else 0)
            prediction.append(1 if p > 0.15 else 0)

        prediction = [x for x in range(len(prediction)) if prediction[x] == 1]
        prediction = " ".join(str(p) for p in prediction)
        print(prediction)

        # q = [x for x in range(len(q)) if q[x] == 1]
        # q = " ".join(str(p) for p in q)
        # print(q)

        writein["Id"].append(imagename[0])
        writein["Predicted"].append(prediction)

        print("=======================================================")

        # if batch_idx != 0 and batch_idx % 10 == 0:
        #     break

    dataframe = pd.DataFrame({'Id':writein["Id"], 'Predicted':writein["Predicted"]})
    dataframe.to_csv(csv_file, index=0)

if __name__ == '__main__':
    print("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-----------------------------------------------'))
    args = get_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.showGPU
    main()

