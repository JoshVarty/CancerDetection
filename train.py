from fastai.vision import *
import pandas as pd
from fastai.metrics import error_rate
from se_resnet import se_resnet50

from sklearn.metrics import roc_auc_score

def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score

    return score

def train(model, epochs):
    print("Training: ", model.__name__)

    path = 'data/'

    tfms = get_transforms(flip_vert=True, max_rotate=180)
    #Remove symmetric warp
    tfms[0].remove(tfms[0][2])

    cancer_stats = ([0.6961362802832489, 0.5437599535115032, 0.7002275522290441], [0.21564218734453175, 0.23916592699846356,0.2822393775558143])
    data = ImageDataBunch.from_csv(path, folder='train', csv_labels='train_labels.csv', suffix='.tif', valid_pct=0.1,
            ds_tfms=tfms, size=96, num_workers=4, bs=128).normalize(imagenet_stats)
    
    learner = create_cnn(data, model, metrics=[accuracy, auc_score])
    learner.fit_one_cycle(5, max_lr=1e-1)
    learner.unfreeze()
    learner.fit_one_cycle(epochs, max_lr=slice(1e-6,1e-2))
    learner.save(model.__name__ + "_saved")
    learner.export(model.__name__)

if __name__ == "__main__":

    #train(models.resnet50, 20)
    #train(models.resnet101, 20)
    #train(models.resnet152, 25)
    train(models.densenet121, 25)
    train(models.densenet161, 25)
    
