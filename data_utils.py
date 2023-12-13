from datasets import load_dataset
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

def get_dataset():
    dataset = load_dataset("sled-umich/Action-Effect")["ActionEffect"]
    return dataset

def get_transform():
    transform = transforms.Compose([ 
        Resize(),
        transforms.PILToTensor(),
        MoreTransforms()
    ]) 
    return transform

class Resize(object):
    def __call__(self, img):
        img = img.resize((224,224))
        return img
class MoreTransforms(object):
    def __call__(self, img):
        img = img.to(torch.float32)
        img = img[None,:,:,:]

        if img.shape[1] == 1:
            img = img.repeat(1,3,1,1)
        if img.shape[1] == 4:
            img = img[:,:3,:,:]
        
        return img

def preprocess():
        # image_list = i["positive_image_list"] + i["negative_image_list"]
        # truth = torch.cat(torch.tensor([0, 1]).repeat(len(i["positive_image_list"]), 1), torch.tensor([1, 0]).repeat(len(i["negative_image_list"]), 1), dim=0)
        # image_batch = 
        pass

def sample_random_image(dataset, num_images):
    out = []
    dataset_inds = list(range(0, len(dataset)))
    
def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
        Args:
            x (tensor): input images
        """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x

  

