class DefaultConfigs(object):
    architecture='bninception'
    showGPU = '0,1'
    useGPU = '0,1'

    folds = 5
    foldnum = 1

    opt = 'adam'
    lr = 0.03
    batchSize = 8
    nEpochs = 500
    step_size = 10
    gamma = 0.2

config = DefaultConfigs()
