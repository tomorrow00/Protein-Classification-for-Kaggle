class DefaultConfigs(object):
    architecture='bninception'
    showGPU = '1'
    useGPU = '0'

    folds = 5
    foldnum = 5

    opt = 'sgd'
    lr = 0.03
    batchSize = 16
    nEpochs = 500
    step_size = 10
    gamma = 0.2

config = DefaultConfigs()
