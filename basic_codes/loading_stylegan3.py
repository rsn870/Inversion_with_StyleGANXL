import io
import os, time, glob
import pickle
import shutil
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import utils
import dnnlib
import legacy


device = torch.device('cuda:0')


def return_model(checkpoint_path='imagenet512.pkl',device='cpu'):
	with dnnlib.util.open_url(checkpoint_path) as f:
	
		G = legacy.load_network_pkl(f)['G_ema'].to(device)
	return G

def generate_class_averages(G):

	zs = torch.randn([10000, G.mapping.z_dim], device=device)
	cs = torch.zeros([10000, G.mapping.c_dim], device=device)
	for i in range(cs.shape[0]):
		cs[i,i//10]=1
	w_stds = G.mapping(zs, cs)

	w_stds = w_stds.reshape(10, 1000, G.num_ws, -1)
	w_stds=w_stds.std(0).mean(0)[0]
	w_all_classes_avg = G.mapping.w_avg.mean(0)

	return w_all_classes_avg,w_stds

def generate_from_w(G,w):
	image = G.synthesis((w).unsqueeze(1).repeat([1, G.num_ws, 1])) #Set up in W+ Space!

def generate_image_from_w_class_norm(G,w,class_mean,class_std):
	image = G.synthesis((w*class_std+class_mean).unsqueeze(1).repeat([1, G.num_ws, 1])) #Assuming appropriate norm


def generate_from_wlist(G,w):
	image=G.synthesis(w) #Assuming the tensor after concat has dims #BSxG_num_wsx512

def save_image(image,f):
	image = image.clamp(-1,1).cpu()
	utils.save_image(image,f)

