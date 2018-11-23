import torch
from torchsummary import summary
from torch.utils.data import DataLoader

# base libraries
import os
import argparse
import numpy as np
import pandas as pd

# internals
from src import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

DATA_DIR = '/home/wcc/data/kaggle_protein'
TESTDATA = os.path.join(DATA_DIR, 'test')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network-name', type=str, required=True)
    parser.add_argument('-d', '--dataset-name', type=str, required=True)
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size') # 64
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()

    args.save = 'work/%s/%s' % (args.network_name, args.dataset_name)

    torch.cuda.manual_seed(args.seed)

    dataset = get_dataset(TESTDATA, False)
    testLoader = DataLoader(dataset=dataset, batch_size=args.batchSize, num_workers=4, pin_memory=True, shuffle=False)

    csv_file = os.path.join(args.save, 'test.csv')
    print(csv_file)

    args.load = sorted(os.listdir(args.save), key=lambda x:int(x.split(".")[0]), reverse=True)[0]
    args.load = os.path.join(args.save, args.load)
    print("Loading network: {}".format(args.load))
    net = torch.load(args.load)
    net = net.cuda()
    summary(net, input_size=(4, 512, 512))
    print('Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    test(net, testLoader, csv_file)

def test(net, testLoader, csv_file):
    net.eval()
    writein = {}
    writein["Id"] = []
    writein["Predicted"] = []

    for batch_idx, data in enumerate(testLoader):
        print(data['image_id'][0])
        inputs, labels = data['image'], data['labels']

        inputs = inputs.cuda()

        outputs = net(inputs)

        pred = outputs.data.gt(0.5)
        prediction = pred[0].cpu().numpy()
        print(prediction)
        # prediction = [np.argwhere(p == 1) for p in prediction]
        prediction = [x for x in range(len(prediction)) if prediction[x] == 1]
        # print(prediction)
        # print(list(map(int, np.where(prediction == 1))))
        prediction = " ".join(str(p) for p in prediction)
        print(prediction)

        writein["Id"].append(data['image_id'][0])
        writein["Predicted"].append(prediction)

        print("==================================================================================")

        if batch_idx != 0 and batch_idx % 10 == 0:
            break

    dataframe = pd.DataFrame({'Id':writein["Id"], 'Predicted':writein["Predicted"]})
    dataframe.to_csv(csv_file, index=0)

if __name__ == '__main__':
    main()