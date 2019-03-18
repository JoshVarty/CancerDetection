import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from fastai.vision import *
from sklearn.metrics import roc_auc_score

wsi = pd.read_csv('data/patch_id_wsi.csv')

lookup = {}
for i, slide_name in enumerate(wsi['wsi']):
    
    slide_id = slide_name.split('_')[-1]
    if slide_id not in lookup:
        lookup[slide_id] = []
        
    imageId = wsi.iloc[i]['id']
    lookup[slide_id].append(imageId)

validationIds = []
lookupIterator = iter(lookup.keys())
while len(validationIds) < 20000: #~10% of training set
    slideId = next(lookupIterator)
    imageIds = lookup[slideId]
    validationIds.extend(imageIds)
    print("Added slide", slideId)


print("Validation Set:", len(validationIds))    

for imageId in tqdm(validationIds):
    fromPath = 'data/train/' + imageId + '.tif'
    toPath = 'data/valid/' + imageId + '.tif'
    shutil.copy(fromPath, toPath)