from fastai.vision import *
import pandas as pd
from fastai.metrics import error_rate
from se_resnet import se_resnet50


def train(model):
    print("Training: ", model.__name__)

    path = 'data/'

    tfms = get_transforms()
    #Remove symmetric warp
    tfms[0].remove(tfms[0][2])

    cancer_stats = [[177.51475147222848, 138.6587881454333, 178.55802581840624], [54.9887577728556, 60.98731138460821, 71.97104127673265]]
    data = ImageDataBunch.from_csv(path, folder='train', csv_labels='train_labels.csv', suffix='.tif', valid_pct=0.1,
            ds_tfms=tfms, size=96, num_workers=4).normalize(cancer_stats)
    
    learner = create_cnn(data, model, metrics=accuracy)
    learner.unfreeze()
    learner.fit_one_cycle(30, max_lr=slice(1e-6,1e-2))
    learner.save(model.__name__ + "_saved")
    learner.export(model.__name__)

if __name__ == "__main__":

    pre_trained_models = [models.resnet34, models.resnet50, models.resnet101, models.densenet121, models.squeezenet1_1]
    for model in pre_trained_models:
        train(model)
    
    untrained_models = [se_resnet50]
    for model in untrained_models:
        train(model)
    
