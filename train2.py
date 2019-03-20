from fastai.vision import *
import pandas as pd
from fastai.metrics import error_rate
from se_resnet import se_resnet50
import torch
from sklearn.metrics import roc_auc_score

def gaussian_kernel(size, sigma=0.3, dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

def _gaussian_blur(x, size:uniform_int):
    kernel = gaussian_kernel(size=size)
    kernel_size = 2*size + 1

    x = x[None,...]
    padding = int((kernel_size - 1) / 2)
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    x = torch.squeeze(F.conv2d(x, kernel, groups=3))

    return x

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

    #Set up transformations
    tfms = get_transforms(flip_vert=True, max_rotate=180, p_affine=0.875, max_warp=0, max_zoom=1.3)
    gaussian_blur = TfmPixel(_gaussian_blur)
    gb = gaussian_blur(size=(1, 3), p=0.2)
    tfms[0].append(gb)

    #Set up databunch
    rawFilenames = os.listdir('data/valid')
    validFilenames = {}
    for filename in rawFilenames:
        imageId = filename.split('.')[0]
        validFilenames[imageId] = 1
        
    def checkIfInValidationSet(fullPath):
        filename = fullPath.split('/')[-1]
        imageId = filename.split('.')[0]
        
        if imageId in validFilenames:
            return True
        else:
            return False
    
    validItems = ImageList.from_folder('data/valid')
    trainItems = ImageList.from_folder('data/train')
    data = (ImageList.from_csv(path, 'train_labels.csv', folder='train', suffix='.tif')
        .split_by_valid_func(checkIfInValidationSet)
        .label_from_df()
        .transform(tfms)
        .databunch(num_workers=4, bs=64)
        .normalize(imagenet_stats))
  
    learner = cnn_learner(data, model, metrics=[accuracy, auc_score])
    #Train on the final layer
    learner.fit_one_cycle(5, max_lr=1e-1)
    learner.unfreeze()

    #One round of cyclical learning
    learner.fit_one_cycle(epochs, max_lr=slice(1e-6,1e-2))
    learner.save('best')
    
    #Learning Rate Decay with Restarts
    best_loss = learner.recorder.val_losses[-1]
    print("Best Loss:", best_loss)
    counter = 0
    learner = learner.load('best')
    learner.unfreeze()

    max_lr = 1e-3
    min_lr = 1e-6

    while True:
        print("~~~~~~~~~~~~~~~~~~~~")
        learner.fit(5, slice(min_lr, max_lr))
    
        last_loss = learner.recorder.val_losses[-1]
        
        if last_loss < best_loss:
            #If we've improved, save that.
            print("Improved. Saving...")
            best_loss = last_loss
            learner.save('best')
        else:
            #Otherwise, halve the learning rate
            print("No improvement. Reloading.")
            learner = learner.load('best')
            learner.unfreeze()
            min_lr = min_lr / 2
            max_lr = max_lr / 2
            print("MinLR:", min_lr)
            print("MaxLR:", max_lr)
        
        if max_lr < 5e-5:
            break

    learner.save(model.__name__ + "_saved")
    learner.export(model.__name__)

if __name__ == "__main__":

    train(models.resnet18, 1)
    #train(models.resnet101, 20)
    #train(models.resnet152, 25)
    #train(models.densenet121, 25)
    #train(models.densenet161, 25)
    
