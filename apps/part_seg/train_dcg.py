import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import torch.optim as optim
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../attack')))

import affordances_loader
import shapenet_part_loader
from model_dcg import DCGPartSeg

from FGM.FGM import FGM,IFGM,MIFGM,PGD
from util.clip_utils import ClipPointsL2
import json 

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

    #create folder to save trained models
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    if opt.dataset=='shapenet_part':
        train_dataset = shapenet_part_loader.PartDataset(classification=False, npoints=opt.num_points, split='train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

        val_dataset = shapenet_part_loader.PartDataset(classification=False, npoints=opt.num_points, split='val')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        
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
        train_dataset = affordances_loader.PartDataset(classification=False, npoints=opt.num_points, split='train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

        val_dataset = affordances_loader.PartDataset(classification=False, npoints=opt.num_points, split='val')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
        
        all_obj_cat_file = os.path.join(dataset_main_path, opt.dataset, 'synsetoffset2category.txt')


    fin = open(all_obj_cat_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    objcats = [line.split()[1] for line in lines]
    objnames = [line.split()[0] for line in lines]
    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()

    writer = SummaryWriter(opt.outf+"/tensorboard")

    for epoch in range(opt.n_epochs):
        if epoch < 50:
            optimizer = optim.Adam(seg_net.parameters(), lr=0.0001)
        elif epoch<150:
            optimizer = optim.Adam(seg_net.parameters(), lr=0.00001)
        else:
            optimizer = optim.Adam(seg_net.parameters(), lr=0.000001)

        # train
        seg_net.train()
        # network.clearScore()
        l1_sum, l2_sum = 0, 0
        iterations = 0

        correct_classified = 0
        total_classified = 0

        correct_classified_print = 0
        total_classified_print = 0
        for batch_id, data in enumerate(train_dataloader):
            points, part_label_, cls_label_ = data
            if(points.size(0)<opt.batch_size):
                break
            points = Variable(points)
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

            optimizer.zero_grad()
            
            if opt.use_tensor_cores:
                with torch.cuda.amp.autocast():
                    pred = seg_net(points_adv,cur_label_one_hot)
                    train_loss = seg_net.module.cross_entropy_loss(pred, part_label)
            else:
                pred = seg_net(points_adv,cur_label_one_hot)
                train_loss = seg_net.module.cross_entropy_loss(pred, part_label)
            
            
            if opt.use_tensor_cores:
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                train_loss.backward()
                optimizer.step()
                
            l1_sum += train_loss.item()
            iterations+=1

            #8, 50, 2048 -> 8, 2048, 50 -> 8*2048, 50

            pred_choice = np.array(pred.transpose(1,2).reshape(-1,pred.shape[1]).data.cpu().max(1)[1])
            ground_truth =  np.array(part_label.view(-1).data.cpu())
            correct = np.sum(pred_choice == ground_truth)
            correct_classified_print += correct
            total_classified_print += len(ground_truth)
            correct_classified += correct
            total_classified += len(ground_truth)

            if batch_id % 50 == 0:
                accuracy_print = correct_classified_print/total_classified_print
                correct_classified_print = 0
                total_classified_print = 0
                print('batch_no:%d/%d, loss: %f accuracy: %f' %  (batch_id, len(train_dataloader), train_loss, accuracy_print), flush=True)
        

        accuracy = correct_classified/total_classified
        writer.add_scalar('loss', l1_sum/iterations, epoch)
        writer.add_scalar('accuracy', accuracy, epoch)
        print('Averages of epoch %d : loss: %f accuracy: %f' %
            (epoch, (l1_sum  / iterations), accuracy))

        with torch.no_grad():
            iterations = 0
            l1_sum = 0
            correct_classified = 0
            total_classified = 0

            for batch_id, data in enumerate(val_dataloader):
                points, part_label_, cls_label_ = data
                if(points.size(0)<opt.batch_size):
                    break
                points = Variable(points)
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
                
                points_adv = points

                
                if opt.use_tensor_cores:
                    with torch.cuda.amp.autocast():
                        pred = seg_net(points_adv,cur_label_one_hot)
                        val_loss = seg_net.module.cross_entropy_loss(pred, part_label)
                else:
                    pred = seg_net(points_adv,cur_label_one_hot)
                    val_loss = seg_net.module.cross_entropy_loss(pred, part_label)
                    
                l1_sum += val_loss.item()
                iterations+=1

                #8, 50, 2048 -> 8, 2048, 50 -> 8*2048, 50

                pred_choice = np.array(pred.transpose(1,2).reshape(-1,pred.shape[1]).data.cpu().max(1)[1])
                ground_truth =  np.array(part_label.view(-1).data.cpu())
                correct = np.sum(pred_choice == ground_truth)
                correct_classified += correct
                total_classified += len(ground_truth)

            accuracy = correct_classified/total_classified
            writer.add_scalar('loss_validation', l1_sum/iterations, epoch)
            writer.add_scalar('accuracy_validation', accuracy, epoch)
            print('Validation of epoch %d : loss: %f accuracy: %f' %
                (epoch, (l1_sum  / iterations), accuracy))
        

        if epoch% 5 == 0:
            dict_name=opt.outf+'/'+opt.dataset+'_dataset_'+str(epoch)+'.pth'
            torch.save(seg_net.module.state_dict(), dict_name)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
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

    opt = parser.parse_args()
    print(opt)

    main()
