from __future__ import  absolute_import
import os

import numpy as np
import matplotlib
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils.config import opt
from data.dataset import inverse_normalize
from data.fetus_dataset import fetus_Dataset, collate_fn
from topograph_net import Topograph

from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from utils.boxlist import BoxList
from utils.gpu_tools import get_world_size, get_global_rank, get_local_rank, get_master_ip

from utils.distributed import get_rank, synchronize, reduce_loss_dict, DistributedSampler, all_gather

import resource
import wandb

import warnings
warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        print('Load Fetus Dataset')
        trainset = fetus_Dataset(self.opt, operation='train')
        self.train_dataloader = DataLoader(trainset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=self.opt.num_workers)

        vaildset = fetus_Dataset(self.opt, operation='valid')
        self.vaild_dataloader = DataLoader(vaildset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,)

        testset  = fetus_Dataset(self.opt, operation='test')
        self.test_dataloader = DataLoader(testset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=self.opt.test_num_workers,
                                        shuffle=False,)

        self.model = Topograph(self.opt).to(device=opt.device)
        print('model construct completed')

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.opt.lr,
                                   momentum=0.9,
                                   nesterov=True,)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, eps=1e-08)
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200], gamma=0.1)

    def train(self):
        if self.opt.load_path:
            self.load(self.opt.load_path)
            print('load pretrained model from %s' % self.opt.load_path)

        best_map = 0
        lr_ = self.opt.lr
        for epoch in range(self.opt.epoch):
            self.model.train()
            for step, (imgs, targets, ids) in enumerate(tqdm(self.train_dataloader)):
                # if step > 1:
                #     break
                imgs = imgs.tensors.to(device=opt.device)
                targets = [target.to(device=opt.device) for target in targets]

                _, losses = self.model(imgs, image_sizes=None, targets=targets, train=True)

                loss_cls = losses['loss_cls'].mean()
                loss_box = losses['loss_box'].mean()
                loss_center = losses['loss_center'].mean()

                loss = loss_cls + loss_box + loss_center
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()

            eval_result = self.eval(self.vaild_dataloader, test_num=self.opt.test_num)
            log_info = 'epoch:{}, map:{},loss:{}'.format(str(epoch),
                                                         str(round(eval_result['map'], 4)),
                                                         str(loss.item()))
            print(log_info)

            if eval_result['map'] > best_map and epoch > 50:
                best_map = eval_result['map']
                best_path = self.save(best_map=best_map)

            # if epoch % 100 == 0 and epoch != 0:
            #     model, optimizer = load(opt, model, optimizer, best_path)
            #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
            #     lr_ = lr_ * opt.lr_decay

            if epoch > 200: 
                self.load(best_path)
                test_result = self.eval(self.test_dataloader, test_num=self.opt.test_num)
                log_info = 'final test ---> epoch:{}, map:{},loss:{}'.format(str(epoch),
                                                                             str(test_result['map']),
                                                                             str(loss.item()))
                print(log_info)
                break

    def accumulate_predictions(self, predictions):
        all_predictions = all_gather(predictions)

        if get_rank() != 0:
            return

        predictions = {}

        for p in all_predictions:
            predictions.update(p)

        ids = list(sorted(predictions.keys()))

        if len(ids) != ids[-1] + 1:
            print('Evaluation results is not contiguous')

        predictions = [predictions[i] for i in ids]

        return predictions

    @torch.no_grad()
    def eval(self, dataloader, test_num=10000):
        self.model.eval()
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        for ids, (imgs, gt_targets, ids) in tqdm(enumerate(dataloader)):

            preds = {}
            imgs = imgs.tensors.to(device=opt.device)

            gt_targets = [target.to('cpu') for target in gt_targets]

            pred, _ = self.model(imgs, imgs.shape[-2:], train=False)
            pred = [p.to('cpu') for p in pred]
            preds = pred

            for idx, pred in enumerate(preds):
                _pred_bboxes = pred.box.numpy()
                _pred_labels = pred.fields['labels'].numpy()
                _pred_scores = pred.fields['scores'].numpy()
                _gt_bboxes_ = gt_targets[idx].box.numpy()
                _gt_labels_ = gt_targets[idx].fields['labels'].numpy()

            if _pred_bboxes.shape[0] == 0:
                continue
            else:
                pred_bboxes += [_pred_bboxes]
                pred_labels += [_pred_labels]
                pred_scores += [_pred_scores]

                gt_bboxes += [_gt_bboxes_]
                gt_labels += [_gt_labels_]
                # gt_difficults.append(gt_difficults_)

            if ids == test_num: break

        gt_difficults = None
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)
        return result

    def save(self, save_optimizer=True, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.model.state_dict()
        save_dict['config'] = opt._state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False,):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.model.load_state_dict(state_dict)
            return self
        if parse_opt:
            self.opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])

def main(rank, opt):

    try:
        opt.local_rank
    except AttributeError:
        opt.global_rank = rank
        opt.local_rank = opt.enable_GPUs_id[rank]
    else:
        if opt.distributed:
            opt.global_rank = rank
            opt.local_rank = opt.enable_GPUs_id[rank]

    if opt.distributed:
        torch.cuda.set_device(int(opt.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt.init_method,
                                             world_size=opt.world_size,
                                             rank=opt.global_rank,
                                             group_name='mtorch'
                                             )

        print('using GPU {}-{} for training'.format(
            int(opt.global_rank), int(opt.local_rank)
            ))

        if opt.local_rank == opt.enable_GPUs_id[0]:
            wandb_init()

    if torch.cuda.is_available(): 
        opt.device = torch.device("cuda:{}".format(opt.local_rank))
    else: 
        opt.device = 'cpu'
    
    Train_ = Trainer(opt)
    Train_.train()

if __name__ == '__main__':

    # setting distributed configurations
    opt.world_size = len(opt.enable_GPUs_id)
    opt.init_method = f"tcp://{get_master_ip()}:{23455}"
    opt.distributed = True if opt.world_size > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and opt.distributed:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=opt.world_size, args=(opt,))
    else:
        # multiple processes have been launched by openmpi
        opt.local_rank = opt.enable_GPUs_id[0]
        opt.global_rank = opt.enable_GPUs_id[0]

        main(opt.local_rank, opt)
