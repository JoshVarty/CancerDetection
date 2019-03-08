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

    cancer_stats = ([0.6961362802832489, 0.5437599535115032, 0.7002275522290441], [0.21564218734453175, 0.23916592699846356,0.2822393775558143])
    data = ImageDataBunch.from_csv(path, folder='train', csv_labels='train_labels.csv', suffix='.tif', valid_pct=0.1,
            ds_tfms=tfms, size=96, num_workers=4).normalize(cancer_stats)
    
    learner = create_cnn(data, model, metrics=accuracy)
    learner.unfreeze()
    learner.fit_one_cycle(15, max_lr=slice(1e-6,1e-2))
    learner.save(model.__name__ + "_saved")
    learner.export(model.__name__)

if __name__ == "__main__":

    pre_trained_models = [models.resnet50, models.resnet101, models.resnet152, models.densenet121]
    for model in pre_trained_models:
        train(model)
    
