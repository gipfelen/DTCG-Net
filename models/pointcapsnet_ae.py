#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:57:49 2018

@author: zhao
"""

from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'nndistance'))
#from modules.nnd import NNDModule
import torch_nndistance as NNDModule
#distChamfer = NNDModule()
distChamfer = NNDModule.NNDFunction.apply
USE_CUDA = True


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        if(output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=64, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64, clustering_fix=True):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size # Output vector per capusle
        self.prim_caps_size = prim_caps_size # Primary point capsule input size
        self.latent_caps_size = latent_caps_size # Latent capsule input size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
        self.clustering_fix = clustering_fix
    
    def t_score(self, c_ij):
        """ Calculates the T-score to measure whether capsules are 
            coupled in a tree structure (~1) or not (~0) [1]
    
            param: c_ij - coupling coefficient with shape (batch_size, out_caps, in_caps, 1) 
        """
        out_caps = float(c_ij.size()[1])
        c_ij = torch.squeeze(c_ij, dim=3)         # (batch_size, out_caps, in_caps) 
        c_ij = c_ij.transpose(2,1)             # (batch_size, in_caps, out_caps)


        epsilon = 1e-12
        entropy = -torch.sum(c_ij * torch.log(c_ij + epsilon), dim=-1)
        T = 1 - entropy / -np.log(1 / out_caps)
        return torch.mean(T).cpu()


    def d_score(self, v_j):
        """ Measures how the activation of capsules adapts to the input.
    
            param: v_j - activations of capsules with shape (batch_size, num_capsules, dim)
        """
        v_j_norm = torch.norm(v_j, dim=-1)
        v_j_std = torch.std(v_j_norm, dim=0, unbiased=False)   # Note: Calc std along the batch dimension
        return torch.max(v_j_std).cpu()

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach() if not self.clustering_fix else u_hat
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)

        v_j = v_j.squeeze(-2)
        c_ij = c_ij.unsqueeze(-1)
        T = self.t_score(c_ij.detach())
        D = self.d_score(v_j.detach())
                
        return v_j, T, D
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class LatentGammaCapsLayer(nn.Module):
     def __init__(self, latent_caps_size=64, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentGammaCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size # Output vector per capusle
        self.prim_caps_size = prim_caps_size # Primary point capsule input size
        self.latent_caps_size = latent_caps_size # Latent capsule input size
        self.latent_vec_size = latent_vec_size
        self.W = nn.Parameter(0.01*torch.randn(1,latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
        self.bias = torch.full((1,latent_caps_size, 1, latent_vec_size), 0.01)
        
     def t_score(self, c_ij):
        """ Calculates the T-score to measure whether capsules are 
            coupled in a tree structure (~1) or not (~0) [1]
    
            param: c_ij - coupling coefficient with shape (batch_size, out_caps, in_caps, 1) 
        """
        out_caps = float(c_ij.size()[1])
        c_ij = torch.squeeze(c_ij, dim=3)         # (batch_size, out_caps, in_caps) 
        c_ij = c_ij.transpose(2,1)             # (batch_size, in_caps, out_caps)


        epsilon = 1e-12
        entropy = -torch.sum(c_ij * torch.log(c_ij + epsilon), dim=-1)
        T = 1 - entropy / -np.log(1 / out_caps)
        return torch.mean(T).cpu()


     def d_score(self, v_j):
        """ Measures how the activation of capsules adapts to the input.
    
            param: v_j - activations of capsules with shape (batch_size, num_capsules, dim)
        """
        v_j_norm = torch.norm(v_j, dim=-1)
        v_j_std = torch.std(v_j_norm, dim=0, unbiased=False)   # Note: Calc std along the batch dimension
        return torch.max(v_j_std).cpu()
    

     def squash(self,vectors, dim=-1):
        epsilon = 1e-12
        vector_squared_norm = torch.sum(torch.mul(vectors, vectors), dim=dim, keepdim=True) + epsilon
        return (vector_squared_norm / (1 + vector_squared_norm)) * (vectors / torch.sqrt(vector_squared_norm)) + epsilon

     def forward(self, x):
        u_norm = torch.norm(x, dim=-1)
        u = torch.unsqueeze(x, -1)
        u = torch.unsqueeze(u, 1)
        batchSize = x.size(0)

        u_hat = torch.matmul(self.W, u)
        u_hat = torch.squeeze(u_hat, dim=-1)

        # Ensure that ||u_hat|| <= ||v_i|| # We use u_i and not v_i
        u_hat_norm = torch.norm(u_hat, dim=-1, keepdim=True)
        u_norm = torch.unsqueeze(u_norm, 1)
        u_norm = torch.unsqueeze(u_norm, -1)
        u_norm = u_norm.repeat(1, u_hat_norm.size(1), 1, 1)
        new_u_hat_norm = torch.min(u_hat_norm, u_norm)
        u_hat = u_hat / u_hat_norm * new_u_hat_norm
        #u_hat_detatch = u_hat.detach()

        # Scaled-distance-agreement routing
        bias = self.bias.repeat(x.size(0),1,1,1).cuda() #torch.Size([4, 1, 64, 64])
        b = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size, 1)).cuda() #print(b_ij.size())

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b, 1) #torch.Size([4, 64, 1024, 1])

            if iteration == num_iterations - 1:
                s_j = torch.sum(c_ij * u_hat, dim=-2, keepdim=True) + bias #torch.Size([4, 64, 1, 64])
                v_j = self.squash(s_j) #torch.Size([4, 64, 1, 64]))

            else:
                s_j = torch.sum(c_ij * u_hat, dim=-2, keepdim=True) + bias #torch.Size([4, 64, 1, 64])
                v_j = self.squash(s_j)


                v_j = v_j.repeat(1,1,c_ij.size(2),1)  # torch.Size([4, 64, 1024, 64])

                p_p = 0.9
                d = torch.norm(v_j - u_hat, dim=-1, keepdim=True)
                d_o = torch.mean(torch.mean(d))
                d_p = d_o * 0.5
                t = torch.tensor(np.log(p_p * ( self.latent_caps_size - 1)) - np.log(1 - p_p), dtype=torch.float32).cuda() / (d_p - d_o + 1e-12)
                t = torch.unsqueeze(t, -1)
                    
                # Calc log prior using inverse distances
                b = t * d

        v_j = v_j.squeeze(-2)
        T = self.t_score(c_ij.detach())
        D = self.d_score(v_j.detach())
                
        return v_j, T, D
        

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size=latent_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/latent_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
    def forward(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.latent_caps_size))
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

    
class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points, n_classes, routing='sda', clustering_fix=True, disable_classification_capsule=False):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.disable_classification_capsule = disable_classification_capsule
        if routing.lower() == 'sda':
            self.latent_caps_layer = LatentGammaCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
            if not self.disable_classification_capsule:
                self.latent_caps_layer2 = LatentGammaCapsLayer(n_classes, latent_caps_size, latent_vec_size, latent_vec_size)
        else:
            self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size, clustering_fix)
            if not self.disable_classification_capsule: 
                self.latent_caps_layer2 = LatentCapsLayer(n_classes, latent_caps_size, latent_vec_size, latent_vec_size, clustering_fix)

        self.caps_decoder = CapsDecoder(latent_caps_size,latent_vec_size, num_points)

        self.n_classes = n_classes
        self.T_List = []
        self.D_List = []
        
    def clearScore(self):
        self.T_List = []
        self.D_List = []
    
    def tScore(self):
        return np.mean(np.array(self.T_List))
    
    def dScore(self):
        return np.mean(np.array(self.D_List))

    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules, T, D = self.latent_caps_layer(x2)
        logits = None
        if not self.disable_classification_capsule:
            latent_capsules2, _, _ = self.latent_caps_layer2(latent_capsules)
            logits = torch.norm(latent_capsules2, dim=-1)

        reconstructions = self.caps_decoder(latent_capsules)

        self.T_List.append(T)
        self.D_List.append(D)

        return logits, latent_capsules, reconstructions

    def loss(self, data, reconstructions, logits, cls_label):
        l1 = self.reconstruction_loss(data, reconstructions)
        l2 = self.cross_entropy_loss(logits, cls_label) if not self.disable_classification_capsule else 0
        return l1,l2
    
    def cross_entropy_loss(self, logits, cls_label):
        cls_label = torch.squeeze(cls_label,-1)
        return F.cross_entropy(logits, cls_label)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 
    
# This is a single network which can decode the point cloud from pre-saved latent capsules
class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points):
        super(PointCapsNetDecoder, self).__init__()
        self.caps_decoder = CapsDecoder(digit_caps_size,digit_vec_size, num_points)
    def forward(self, latent_capsules):
        reconstructions = self.caps_decoder(latent_capsules)
        return  reconstructions

if __name__ == '__main__':
    USE_CUDA = True
    batch_size=8
    
    prim_caps_size=1024
    prim_vec_size=16
    
    latent_caps_size=32
    latent_vec_size=16
    
    num_points=2048

    point_caps_ae = PointCapsNet(prim_caps_size,prim_vec_size,latent_caps_size,latent_vec_size,num_points)
    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()
    
    rand_data=torch.rand(batch_size,num_points, 3) 
    rand_data = Variable(rand_data)
    rand_data = rand_data.transpose(2, 1)
    rand_data=rand_data.cuda()
    
    codewords,reconstruction=point_caps_ae(rand_data)
   
    rand_data_ = rand_data.transpose(2, 1).contiguous()
    reconstruction_ = reconstruction.transpose(2, 1).contiguous()

    dist1, dist2 = distChamfer(rand_data_, reconstruction_)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    print(loss.item())
