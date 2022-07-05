'''
Authors: 
    Dena Bazazian
    Dhananjay Nahata

Contacts: 
    dena.bazazian@cttc.es
    f2015812p@alumni.bits-pilani.ac.in

Modified:
    Time: 5 Aug 2020 
'''

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

def cov3D_eigen(feature,device): 
    '''
    features --> indices of k nearest neighbors [B,N,k,D]  (64, 6, 1024, 20)  -> (0, 2, 3, 1)
    '''
    k = feature.size(2)
    mean = torch.mean(feature, dim=2) #mean of neighbors along euclidean dimensions
    mean = mean.unsqueeze(2).repeat(1,1,k,1)
    diffs = torch.sub(feature, mean)

    cov = torch.div(torch.einsum('tijk,tijl->tikl', diffs, diffs),feature.size(2))
    cov= cov.cpu().detach().numpy()

    _, sv, _ = np.linalg.svd(cov)
    eigen_val = torch.Tensor(sv**0.5).to(device)

    return eigen_val

def geometry_features(eigen):
    # eigens are the 3 eigenvalues of each point in the point cloud   ---> [B,N,3] => [64,1024,3]
    #The orders are eigenvalues are as: eigen[:,:,0] > eigen[:,:,1] > eigen[:,:,2]
    # lambda_1 > lambda_2 > lambda_3
    # lambda_1 = eigen[:,:,0]  , Lambda_2 = eigen[:,:,1]   , Lambda_3 = eigen[:,:,2]

    sum_eigens = ((torch.sum(eigen, 2)).unsqueeze(2))  #summation of all the eigenvalues [B,N]  => [64, 1024]
    omnivariance =  ((torch.pow((torch.prod(eigen,2)),1.0/3.0)).unsqueeze(2))  # third square root of the product (multiplication) over all the three eigenvalues
    eigenentropy = -1*(torch.sum(torch.mul(eigen,(torch.log(eigen))), 2)).unsqueeze(2) #negative summation of the multiplication of each eigenvalue with its ln      
    linearity = torch.div(torch.sub(eigen[:,:,0],eigen[:,:,1]), eigen[:,:,0]).unsqueeze(2)  # (lambda_1 - lambda_2)/lambda_1  => [B,N]
    planarity = torch.div(torch.sub(eigen[:,:,1],eigen[:,:,2]), eigen[:,:,0]).unsqueeze(2)  # (lamda_2 - lambda_3)/Lambda_1  => [B,N]
    sphericity = torch.div(eigen[:,:,2], eigen[:,:,0]).unsqueeze(2)  # (lambda_3)/Lambda_1  => [B,N]
    change_of_curvature = torch.div(eigen[:,:,2], sum_eigens.squeeze(2)).unsqueeze(2)  # (lamnbda_3)/ (lambda_1 + lambda_2 + lambda_3)  => [B,N]
    edges = (eigen[:,:,2]).unsqueeze(2)  # lambda_3  => [B,N]

    geo_feats = torch.cat((sum_eigens, omnivariance, eigenentropy, linearity, planarity, sphericity, change_of_curvature, edges), axis = 2)
    NotNAN_geo_feats = torch.where(torch.isnan(geo_feats), torch.zeros_like(geo_feats), geo_feats)

    return NotNAN_geo_feats  #geo_feats

def GroupLayer(x1, number_of_neighbors, batch_size, num_dims, num_points, idx=None, concat=False):
    feature = x1.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, number_of_neighbors, num_dims)
    if concat:
        x1 = x1.view(batch_size, num_points, 1, num_dims).repeat(1, 1, number_of_neighbors, 1)
        
        feature = torch.cat((feature-x1, feature), dim=3).permute(0, 3, 1, 2)
    return feature

class CapsGraph(nn.Module):
    def __init__(self, device, num_input_capsules, num_secondary_capsules):
        super(CapsGraph, self).__init__()
        self.num_input_capsules = num_input_capsules
        self.num_secondary_capsules = num_secondary_capsules
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        
        u_hat = torch.matmul(x.transpose(2,1), x)
        b_ij = Variable(torch.zeros(batch_size, self.num_secondary_capsules, self.num_input_capsules, device = self.device))

        num_iterations = 3

        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim = 1)
            s_j = (c_ij * u_hat)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                u_ij = torch.sum(v_j * u_hat, dim=2).unsqueeze(2)
                b_ij = b_ij + u_ij

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class GammaCapsGraph(nn.Module):
    def __init__(self, device, num_input_capsules, num_secondary_capsules):
        super(GammaCapsGraph, self).__init__()
        self.num_input_capsules = num_input_capsules
        self.num_secondary_capsules = num_secondary_capsules
        self.device = device
        self.bias = torch.full((1,num_input_capsules, num_secondary_capsules), 0.01)

    def forward(self, x):
        #x: torch.Size([32, 3, 2048])
        batch_size = x.size(0)
        u_norm = torch.norm(x, dim=1)

        # original point cloud:
        # torch.Size([8, 64, 1024, 64])
        # Now:
        # torch.Size([32, 2048, 2048])
        u_hat = torch.matmul(x.transpose(2,1), x) 
                                                  
        # Ensure that ||u_hat|| <= ||v_i|| # We use u_i and not v_i
        u_hat_norm = torch.norm(u_hat, dim=-1, keepdim=True)
        u_norm = torch.unsqueeze(u_norm, -1)

        # original point cloud:
        # torch.Size([8, 64, 1024, 1]) torch.Size([8, 64, 1024, 1])
        # Now:
        # torch.Size([32, 2048, 1]) torch.Size([32, 2048, 1])
        new_u_hat_norm = torch.min(u_hat_norm, u_norm)
        u_hat = u_hat / u_hat_norm * new_u_hat_norm


        b_ij = Variable(torch.zeros(batch_size, self.num_secondary_capsules, self.num_input_capsules, device = self.device))
        bias = self.bias.repeat(x.size(0),1,1).cuda()
    
        num_iterations = 3

        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim = 1)
            s_j = (c_ij * u_hat) + bias
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                u_ij = torch.sum(v_j * u_hat, dim=2).unsqueeze(2)
                b_ij = b_ij + u_ij

                # # new 
                # v_j = v_j.repeat(1,1,c_ij.size(2),1)  # torch.Size([4, 64, 1024, 64])

                p_p = 0.9
                d = torch.norm(v_j - u_hat, dim=-1, keepdim=True)
                d_o = torch.mean(torch.mean(d))
                d_p = d_o * 0.5
                t = torch.tensor(np.log(p_p * ( self.num_secondary_capsules - 1)) - np.log(1 - p_p), dtype=torch.float32).cuda() / (d_p - d_o + 1e-12)
                t = torch.unsqueeze(t, -1)

                    
                # # Calc log prior using inverse distances
                b_ij = t * d

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

def features_extraction(x, number_of_neighbors, device, use_sda=True, concat=False):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    
    if use_sda:
        n2 = GammaCapsGraph(device = device, num_input_capsules = num_points, num_secondary_capsules = num_points)
    else:
        n2 = CapsGraph(device = device, num_input_capsules = num_points, num_secondary_capsules = num_points)


    out1 = n2(x)    
    idx = out1.topk(k=number_of_neighbors, dim=-1)[1]
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    features = GroupLayer(x.transpose(2, 1).contiguous(), number_of_neighbors, batch_size, num_dims, num_points, idx = idx, concat=concat)

    return features

def eigen_features_extraction(x, number_of_neighbors, device, use_sda=True):
    features = features_extraction(x, number_of_neighbors, device, use_sda=use_sda, concat=False)
    eigen = cov3D_eigen(features, device)
    return features_extraction(eigen.transpose(2,1), number_of_neighbors, device, use_sda=use_sda, concat=True)

def geometrical_features_extraction(x, number_of_neighbors, device, use_sda=True):
    features = features_extraction(x, number_of_neighbors, device, use_sda=use_sda, concat=False)
    eigen = cov3D_eigen(features, device)
    geometrical_features = geometry_features(eigen)
    return features_extraction(geometrical_features.transpose(2,1), number_of_neighbors, device, use_sda=use_sda, concat=True)

def euclidean_eigen_features_extraction(x, number_of_neighbors, device, use_sda=True):
    features = features_extraction(x, number_of_neighbors, device, use_sda=use_sda, concat=False)
    eigen = cov3D_eigen(features, device)
    x = torch.cat((x, eigen.transpose(2,1)), axis = 1)
    return features_extraction(x, number_of_neighbors, device, use_sda=use_sda, concat=True)

def euclidean_eigen_Geometrical_features_extraction(x, number_of_neighbors, device, use_sda=True):
    features = features_extraction(x, number_of_neighbors, device, use_sda=use_sda, concat=False)
    eigen = cov3D_eigen(features, device)
    geometrical_features = geometry_features(eigen)
    x = torch.cat((x, eigen.transpose(2,1), geometrical_features.transpose(2,1)), axis = 1)
    return features_extraction(x, number_of_neighbors, device, use_sda=use_sda, concat=True)


class DCGPartSeg(nn.Module):
    def __init__(self, device, seg_num_all=50, features_extraction_method=0, dropout_rate = 0.5, emb_dims = 64, number_of_neighbors = 20, use_sda=True, use_category=True):
        super(DCGPartSeg, self).__init__()
        self.emb_dims = emb_dims
        self.seg_num_all = seg_num_all
        self.number_of_neighbors = number_of_neighbors
        self.features_extraction_method = features_extraction_method
        self.device = device
        self.use_sda = use_sda
        self.use_category = use_category
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        if (self.features_extraction_method == 1 or self.features_extraction_method == 2):
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            
        elif (self.features_extraction_method == 3):
            self.conv1 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
        elif (self.features_extraction_method == 4):
            self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(28, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))


        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280 if self.use_category else 1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout_rate)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
    
    def cross_entropy_loss(self, pred, part_label):
        pred = torch.squeeze(pred,-1)
        part_label = torch.squeeze(part_label,-1)
        return F.cross_entropy(pred, part_label)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        if (self.features_extraction_method == 1):
            x = features_extraction(x, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda, concat=True)    # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        elif (self.features_extraction_method == 2):
            x = eigen_features_extraction(x, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda)    # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        elif (self.features_extraction_method == 3):
            x = geometrical_features_extraction(x, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda)   # (batch_size, 8, num_points) -> (batch_size, 8*2, num_points, k)
        elif (self.features_extraction_method == 4):
            x = euclidean_eigen_features_extraction(x, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda)   # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        else:
            x = euclidean_eigen_Geometrical_features_extraction(x, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda)   # (batch_size, 14, num_points) -> (batch_size, 14*2, num_points, k)
     

        x = self.conv1(x)                       # (batch_size, x.shape[0], num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = features_extraction(x1, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda, concat=True)       # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = features_extraction(x2, number_of_neighbors=self.number_of_neighbors, device = self.device, use_sda=self.use_sda, concat=True)       # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        if self.use_category:
            l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
            l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

            x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x