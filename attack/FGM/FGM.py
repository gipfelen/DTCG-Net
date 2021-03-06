"""Implementation of gradient based attack methods, FGM, I-FGM, MI-FGM, PGD, etc.
Related paper: CVPR'20 GvG-P,
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Self-Robust_3D_Point_Recognition_via_Gather-Vector_Guidance_CVPR_2020_paper.pdf
"""

import torch
import numpy as np


class FGM:
    """Class for FGM attack.
    """

    def __init__(self, model, adv_func, budget,
                 dist_metric='l2'):
        """FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGM attack
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        #self.model.eval()

        self.adv_func = adv_func
        self.budget = budget
        self.dist_metric = dist_metric.lower()

    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        # use global l2 norm here!
        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data, part_label, cls_label=None, normalize=True):
        """Generate one step gradient.

        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            normalize (bool, optional): whether l2 normalize grad. Defaults to True.
        """
        data.requires_grad_()


        # forward pass
        logits = self.model(data,cls_label)

        # backward pass
        loss = self.adv_func(logits, part_label).mean() #unsure about mean
        loss.backward() #unsure
        
        with torch.no_grad():
            grad = data.grad.detach()  # [B, 3, K]
            if normalize:
                norm = self.get_norm(grad)
                grad = grad / (norm[:, None, None] + 1e-9)
        return grad

    def attack(self, data, part_label,cls_label=None):
        """One step FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
        """
        pc = data.clone().detach()

        # gradient
        normalized_grad = self.get_gradient(pc, part_label,cls_label=cls_label)  # [B, 3, K]
        perturbation = normalized_grad * self.budget
        with torch.no_grad():
            #perturbation = perturbation.transpose(1, 2).contiguous()
            data = data - perturbation  # no need to clip

        torch.cuda.empty_cache()
        return data


class IFGM(FGM):
    """Class for I-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2'):
        """Iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(IFGM, self).__init__(model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter

    def attack(self, data, part_label,cls_label=None):
        """Iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, _, K = data.shape
        pc = data.clone().detach()
        randomOffset = torch.randn((B, 3, K)) * 1e-7
        if torch.cuda.is_available():
            randomOffset = randomOffset.cuda()
        pc = pc + randomOffset
        ori_pc = pc.clone().detach()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            normalized_grad = self.get_gradient(pc, part_label,cls_label=cls_label)

            perturbation = self.step_size * normalized_grad

            # add perturbation and clip
            with torch.no_grad():
                pc = pc + perturbation
                pc = self.clip_func(pc, ori_pc)

        return pc


class MIFGM(FGM):
    """Class for MI-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, mu=1.,
                 dist_metric='l2'):
        """Momentum enhanced iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            mu (float): momentum factor
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(MIFGM, self).__init__(model, adv_func,
                                    budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.mu = mu

    def attack(self, data, target, cls_label=None):
        """Momentum enhanced iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()
        momentum_g = torch.tensor(0.).cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            grad, pred = self.get_gradient(pc, target, cls_label=cls_label, normalize=False)
            success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()

            # grad is [B, 3, K]
            # normalized by l1 norm
            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])  # [B]
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self.mu * momentum_g + normalized_grad
            g_norm = self.get_norm(momentum_g)
            normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g

            # add perturbation and clip
            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), \
            success_num


class PGD(IFGM):
    """Class for PGD attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2'):
        """PGD attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(PGD, self).__init__(model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric)

    def attack(self, data, part_label, cls_label=None):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        epsilon = self.budget / \
            ((data.shape[2] * data.shape[1]) ** 0.5)
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation

        dataAttack = super(PGD, self).attack(init_data, part_label, cls_label=cls_label)

        return dataAttack
