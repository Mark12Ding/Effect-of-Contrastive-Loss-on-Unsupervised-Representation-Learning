#!pip install torchsummary
import pickle
import numpy as np
#import matplotlib.pyplot as plt
import os
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import torchvision
from torchvision import datasets, models, transforms

from torchsummary import summary
from tqdm import tqdm_notebook
from model import *
from loss import *
from dataset import *
from train import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losstype",
                    default="spring",
                    help="[spring, triplet, softmax, infonce]")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    args = parser.parse_args()  
    dataloader_un, dataloader_tr, dataloader_te = dataloader()
    criterion_l = Loss().cuda()
    criterion_ab = Loss().cuda()
    K = 127
    if args.losstype == "spring":
        contrast = CMCScore_spring(feat_dim, 100000, K).cuda()
    elif args.losstype == "triplet":
        contrast = CMCScore_triplet(feat_dim, 100000, K).cuda()
    elif args.losstype == "softmax":
        contrast = CMCScore_softmax(feat_dim, 100000, K).cuda()
        criterion_l = Loss(True).cuda()
        criterion_ab = Loss(True).cuda()
    elif args.losstype == "infonce":
        contrast = CMCScore_infonce(feat_dim, 100000, K).cuda()
        criterion_l = NCECriterion(100000).cuda()
        criterion_ab = NCECriterion(100000).cuda()
    encoder_cmc = EncoderCMC().to(device)
    encoder_cmc = train_cmc(encoder_cmc, contrast, criterion_l, criterion_ab, dataloader_un, epochs=100)
    torch.save(encoder_cmc.state_dict(), str(args.losstype)+'.pt')
    encoder_cmc_cat = EncoderCMC_Cat(encoder_cmc)
    linear_cls = nn.Sequential(nn.Linear(feat_dim*2, 10)).to(device)
    cls_cmc, loss_traj_cmc = train_classfier(encoder_cmc_cat, linear_cls, dataloader_tr, epochs=100, supervised=False)
    test(encoder_cmc_cat, cls_cmc, dataloader_te)

if __name__ == '__main__':
    main()
