from open3d import *
from open3d.open3d_pybind.geometry import *
from open3d.open3d_pybind.utility import *
from open3d.open3d_pybind.visualization import *
import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../attack')))

import affordances_loader
import kitchen_data_subset_loader
import shapenet_part_loader
from model_dcg import DCGPartSeg

from FGM.FGM import FGM,IFGM,MIFGM,PGD
from util.clip_utils import ClipPointsL2
import json
import matplotlib.pyplot as plt


def main():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg_net = DCGPartSeg(device,seg_num_all=opt.n_part_classes, features_extraction_method=4, dropout_rate = 0.5, emb_dims = 1024, number_of_neighbors = 20, use_sda=opt.routing == "sda", use_category=(not opt.dont_use_category))

    if opt.model != '':
        seg_net.load_state_dict(torch.load(opt.model))
    
    seg_net = torch.nn.DataParallel(seg_net)
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        seg_net.to(device)
        if opt.use_tensor_cores:
            scaler = torch.cuda.amp.GradScaler()

    if opt.attack != 'None':
        budget = opt.attack_budget
        num_iter = opt.attack_iter
        clip_func = ClipPointsL2(budget=budget)
        step_size = budget / float(num_iter)

        if opt.attack == 'mifgm':
            attacker = MIFGM(seg_net, adv_func=seg_net.module.cross_entropy_loss,
                            clip_func=clip_func, budget=budget, step_size=step_size,
                            num_iter=num_iter, dist_metric='l2')
        else:
            attacker = PGD(seg_net, adv_func=seg_net.module.cross_entropy_loss,
                            clip_func=clip_func, budget=budget, step_size=step_size,
                            num_iter=num_iter, dist_metric='l2')
    
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    if opt.dataset=='shapenet_part':
        test_dataset = shapenet_part_loader.PartDataset(classification=False, npoints=opt.num_points, split='test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

        oid2cpid_file_name=os.path.join(dataset_main_path, opt.dataset,'shapenetcore_partanno_segmentation_benchmark_v0/shapenet_part_overallid_to_catid_partid.json')        
        oid2cpid = json.load(open(oid2cpid_file_name, 'r'))   
        object2setofoid = {}
        for idx in range(len(oid2cpid)):
            objid, pid = oid2cpid[idx]
            if not objid in object2setofoid.keys():
                object2setofoid[objid] = []
            object2setofoid[objid].append(idx)

        all_obj_cat_file = os.path.join(dataset_main_path, opt.dataset, 'shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt')
    elif opt.dataset=='affordances':
        test_dataset = affordances_loader.PartDataset(classification=False, npoints=opt.num_points, split='test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

        all_obj_cat_file = os.path.join(dataset_main_path, opt.dataset, 'synsetoffset2category.txt')
    elif opt.dataset=='kitchen':
        test_dataset = kitchen_data_subset_loader.PartDataset(classification=False, npoints=opt.num_points, split='test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

        all_obj_cat_file = os.path.join(dataset_main_path, "kitchen-data-subset_preprocessed", 'synsetoffset2category.txt')



    fin = open(all_obj_cat_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    objcats = [line.split()[1] for line in lines]
    objnames = [line.split()[0] for line in lines]
    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()

    colors = plt.cm.tab10((np.arange(10)).astype(int))
    pcd_colored = PointCloud()                   
    pcd_ori_colored = PointCloud()
    
    rotation_angle=-np.pi/4
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)           
    flip_transforms  = [[cosval, 0, sinval,-1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    flip_transformt  = [[cosval, 0, sinval,1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    

    iterations = 0
    l1_sum = 0
    correct_classified = 0
    total_classified = 0

    for batch_id, data in enumerate(test_dataloader):
        points_, part_label_, cls_label_ = data
        # if(points_.size(0)<opt.batch_size):
        #     break
        points = Variable(points_)
        points = points.transpose(2, 1)
        if USE_CUDA:
            points = points.cuda()


        if opt.dataset == "shapenet_part":
            for i in range(opt.batch_size):
                iou_oids = object2setofoid[objcats[cls_label_[i]]]
                for j in range(opt.num_points):
                    part_label_[i,j]=iou_oids[part_label_[i,j]]
                        
        cls_label = Variable(cls_label_)
        part_label = Variable(part_label_)

        cur_label_one_hot = np.zeros((opt.batch_size, opt.n_classes), dtype=np.float32)
        for i in range(opt.batch_size):
            cur_label_one_hot[i, cls_label[i]] = 1
        cur_label_one_hot=torch.from_numpy(cur_label_one_hot).float()

        if USE_CUDA:
            cls_label = cls_label.cuda()
            part_label = part_label.cuda()
                
        points_adv = attacker.attack(points,part_label,cur_label_one_hot) if opt.attack != 'None' else points

                
        if opt.use_tensor_cores:
            with torch.cuda.amp.autocast():
                pred = seg_net(points_adv,cur_label_one_hot)
                #val_loss = seg_net.module.cross_entropy_loss(pred, part_label)
        else:
            pred = seg_net(points_adv,cur_label_one_hot)
            #val_loss = seg_net.module.cross_entropy_loss(pred, part_label)
                    
        #l1_sum += val_loss.item()
        iterations+=1

        pred_choice = np.array(pred.transpose(1,2).reshape(-1,pred.shape[1]).data.cpu().max(1)[1])
        #ground_truth =  np.array(part_label.view(-1).data.cpu())
        #correct = np.sum(pred_choice == ground_truth)
        #correct_classified += correct
        #total_classified += len(ground_truth)

        # viz the part segmentation
        if not opt.disable_visualization:
            point_color=torch.zeros([opt.batch_size,opt.num_points,3])
            point_ori_color=torch.zeros([opt.batch_size,opt.num_points,3])

            for point_set_no in range (opt.batch_size):
                if opt.dataset == "shapenet_part":
                    iou_oids = object2setofoid[objcats[cls_label[point_set_no ]]]
                
                pred_choice = pred_choice.reshape((points_.size(0), -1))
                for point_id in range (opt.num_points):
                    part_no=pred_choice[point_set_no,point_id]
                    if opt.dataset == "shapenet_part":
                        part_no-=iou_oids[0]
                        if part_no < 0 or part_no >= len(iou_oids):
                            point_color[point_set_no,point_id,0]=0.8
                            point_color[point_set_no,point_id,1]=0.8
                            point_color[point_set_no,point_id,2]=0.8
                    else:
                        point_color[point_set_no,point_id,0]=colors[part_no,0]
                        point_color[point_set_no,point_id,1]=colors[part_no,1]
                        point_color[point_set_no,point_id,2]=colors[part_no,2]
                    
                pcd_colored.points=Vector3dVector(points_[point_set_no,])
                pcd_colored.colors=Vector3dVector(point_color[point_set_no,])

                
                # for point_id in range (opt.num_points):
                #     part_no=part_label[point_set_no,point_id]
                #     if opt.dataset == "shapenet_part":
                #         part_no-=iou_oids[0]
                #     point_ori_color[point_set_no,point_id,0]=colors[part_no,0]
                #     point_ori_color[point_set_no,point_id,1]=colors[part_no,1]
                #     point_ori_color[point_set_no,point_id,2]=colors[part_no,2]
                    
                # pcd_ori_colored.points=Vector3dVector(points_[point_set_no,])
                # pcd_ori_colored.colors=Vector3dVector(point_ori_color[point_set_no,])
            
                # pcd_ori_colored.transform(flip_transforms)# tansform the pcd in order to viz both point cloud
                pcd_colored.transform(flip_transformt)
                draw_geometries([pcd_colored])
                
    #accuracy = correct_classified/total_classified
    #print('Validation of epoch %d : loss: %f accuracy: %f' %
    #        (epoch, (l1_sum  / iterations), accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset')

    parser.add_argument('--n_classes', type=int, default=16, help='Number of classes')
    parser.add_argument('--n_part_classes', type=int, default=50, help='part classes in all the catagories')

    parser.add_argument('--routing', type=str, default='sda', help='Routing used for capsules. Possible options are sda or rba')

    parser.add_argument('--attack', type=str, default='None', help='Attack to perform')
    parser.add_argument('--attack_budget', type=float, default=0.08, help='Budget for attack')
    parser.add_argument('--attack_iter', type=int, default=50, help='Number of iterations for attack')
    
    parser.add_argument("--use_tensor_cores", type=bool, default=False, help='Enable calculation for tensor cores on RTX2000/3000 series')
    parser.add_argument("--dont_use_category", type=bool, default=False, help="Don't use categories labels to infer segmentation labels")
    parser.add_argument('--disable_visualization', type=bool, default=False, help='Disables visualization')

    opt = parser.parse_args()
    print(opt)

    main()
