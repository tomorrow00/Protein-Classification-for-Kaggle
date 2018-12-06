import torch
from torchsummary import summary
from torch.utils.data import DataLoader

# base libraries
import os
import argparse
import numpy as np
import pandas as pd
import datetime

# internals
from src import *

NUM_ALL = 31073
PROTEIN_NUM = [12884, 1253, 3620, 1560, 1857, 2512, 1007,
               2821, 52, 44, 27, 1092, 687, 536,
               1065, 20, 529, 209, 901, 1481, 171,
               3776, 801, 2964, 321, 8227, 327, 10]

DATA_DIR = os.path.join(os.path.expanduser('~'), 'data/kaggle_protein')

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')

    parser.add_argument('--showGPU', type=str, default="0", help='GPUs for CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--useGPU', type=str, default="0", help='GPUs to use.')

    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    # print(args)

    return args

def main():
    args.save = 'work/%s/%s' % (args.network_name, args.dataset_name)

    ################################ device ##############################

    print("\nCreating architecture...")
    torch.cuda.manual_seed_all(args.seed)

    args.load = sorted(os.listdir(args.save), key=lambda x:int(x.split(".")[0]), reverse=True)[0]
    args.load = os.path.join(args.save, args.load)
    print("Loading network: {}".format(args.load))
    net = torch.load(args.load)
    net = net.cuda()

    device_ids = list(map(int, args.useGPU.split(',')))
    net = torch.nn.DataParallel(net, device_ids=device_ids)

    # summary(net, input_size=(4, 512, 512))
    summary(net, input_size=(3, 512, 512))

    ################################ data ##############################

    print("\nLoading data...")
    dataset = get_dataset(DATA_DIR, mode='test')
    testLoader = DataLoader(dataset=dataset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=False)

    csv_file = os.path.join('result', 'test_' + args.network_name + '_' + str(datetime.datetime.now().strftime("%m-%d %H:%M")) + '.csv')
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
        for i, p in enumerate(pred):
            # print(p, float(PROTEIN_NUM[i] / NUM_ALL))
            prediction.append(1 if p > float(PROTEIN_NUM[i] / NUM_ALL) else 0)
            # prediction.append(1 if p > 0.4 else 0)
        # print(prediction)
        prediction = [x for x in range(len(prediction)) if prediction[x] == 1]
        prediction = " ".join(str(p) for p in prediction)
        print(prediction)

        writein["Id"].append(imagename[0])
        writein["Predicted"].append(prediction)

        print("=======================================================")

        # if batch_idx != 0 and batch_idx % 10 == 0:
        #     break

    dataframe = pd.DataFrame({'Id':writein["Id"], 'Predicted':writein["Predicted"]})
    dataframe.to_csv(csv_file, index=0)

if __name__ == '__main__':
    args = get_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.showGPU
    main()
