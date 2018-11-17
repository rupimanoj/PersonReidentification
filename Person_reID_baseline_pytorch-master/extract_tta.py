import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net,ft_net_dense
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

data_dir = "../data/Market/int/val"
gallery_dir_path = "../data/Market/int/val/gallery"
query_dir_path = "../data/Market/int/val/query"
log_dir = "../data/Market/extracted"

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--model_name', default='dense50', type=str, help='enter saved model name')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )

opt = parser.parse_args()
gpu_ids = []
str_ids = opt.gpu_ids.split(',')
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

data_transforms_query = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.TenCrop((256, 128)),
        #transforms.Scale(256),
        #transforms.Resize((288,144), interpolation=3),
        #transforms.Lambda(lambda crops: torch.stack([transforms.Resize((288,144), interpolation=3)(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])) ])

data_transforms_gallery = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.TenCrop((288, 144)),
        #transforms.Scale(256),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])) ])

def load_network(network):
    save_path = os.path.join('./model',opt.model_name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
	
model_structure = ft_net_dense(1503)
model = load_network(model_structure)
model = model.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
        model = model.cuda()

class Dataset(Dataset):
    def __init__(self, path, transform):
        self.dir = path
        self.image = [f for f in os.listdir(self.dir) if f.endswith('png')]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = self.image[idx]
        img = Image.open(os.path.join(self.dir, name))
        img = self.transform(img)
        #ten_images = transforms.TenCrop(img, 144, 144)
        return {'name': name.replace('.png', ''), 'img': img}


	
def extractor(model, dataloader, data_type):
    def fliplr(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip
    test_names = []
    test_features = torch.FloatTensor()
    weights = [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]
    if data_type=="query":
        weights = [0.1,0.1,0.1,0.1,0.6,0.1,0.1,0.1,0.1,0.6]
    for i_batch, sample in enumerate(dataloader):
        names, images = sample['name'], sample['img']
        print("------------------",images.size())
        images = images.permute(1,0,2,3,4)
        print("------------------",images.size())
        crops, n, c, h, w = images.size()
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()
        index = 0
        for image in images:
            ff = ff +  (model(Variable(image.cuda(), volatile=True))[1].data.cpu())
            index += 1
        ff_norm = torch.norm(ff, p = 2, dim = 1, keepdim = True)
        ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))
        test_names = test_names + names
        test_features = torch.cat((test_features, ff), 0)
    return test_names, test_features


image_datasets = {'val':{'gallery': Dataset(gallery_dir_path, data_transforms_gallery),
					'query': Dataset(query_dir_path, data_transforms_query)}}
dataloaders = {'val':{x: torch.utils.data.DataLoader(image_datasets['val'][x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}}

for dataset in ['val']:
    for subset in ['query', 'gallery']:
        test_names, test_features = extractor(model, dataloaders[dataset][subset], subset)
        results = {'names': test_names, 'features': test_features.numpy()}
        scipy.io.savemat(os.path.join(log_dir, 'feature_%s_%s.mat' % (dataset, subset)), results)
