from fastai.vision import ImageImageList, load_learner, models, DatasetType
import pandas as pd
from fastai.metrics import error_rate
import os


def createSubmission(model):
    path = 'data/'

    with open('data/'+ model.__name__ + '.csv', 'w') as writer:
        writer.write('id,label' + os.linesep)

    data = ImageImageList.from_folder(path + "/test")
    learner = load_learner(path, fname=model.__name__, test=data) 

    preds,y = learner.get_preds(ds_type=DatasetType.Test)

    with open('data/'+ model.__name__ + '.csv', 'a') as writer:
        print("Number of predictions: ", len(preds))
        
        for i in range(len(preds)):
            prediction = preds[i][1].numpy()
            id = data.items[i].name.split('.')[0]
            writer.write(id)
            writer.write(",")
            writer.write(str(prediction))
            writer.write(os.linesep)


if __name__ == "__main__":
    pre_trained_models = [models.resnet34, models.resnet50, models.resnet101, models.resnet152, models.densenet121, models.squeezenet1_1]
    for model in pre_trained_models:
        createSubmission(model)



    
