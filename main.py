import os
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import numpy as np
from torch.autograd import Variable
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

from datasets.cityscapes_Dataset import City_Dataset, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset
from perturbations.augmentations import augment, get_augmentation
from perturbations.fourier import fourier_mix
from perturbations.cutmix import cutmix_combine
from models import get_model
from models.ema import EMA
from utils.eval import Eval, synthia_set_16, synthia_set_13

from utils.util import MemoryQueue
from utils.lib import seed_everything, sinkhorn, ubot_CCD, adaptive_filling
import ot
from PIL import Image


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)



def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
  
def weight_norm(w):
    norm = w.norm(p=2, dim=1, keepdim=True)
    w = w.div(norm.expand_as(w))
    return w  
class Trainer():
    def __init__(self, cfg, logger, writer):

        # Args
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.logger = logger
        self.writer = writer

        # Counters
        self.epoch = 0
        self.iter = 0
        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0

        # Metrics
        self.evaluator = Eval(self.cfg.data.num_classes+1)

        # Loss
        self.ignore_index = -1
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        print('dddd',torch.cuda.device_count())
        # Model
        self.model, params = get_model(self.cfg)
        self.model = nn.DataParallel(self.model)  # TODO: test multi-gpu
        self.model.to(self.device)

        # EMA
        self.ema = EMA(self.model, self.cfg.ema_decay)

        self.k=cfg.k
        # UniOT
        self.num_classes=cfg.data.num_classes
        self.prototypes = Variable(torch.zeros([cfg.data.num_classes, 256]), requires_grad = True)
        
    
        self.target_prototypes = Variable(torch.randn([self.k, 256]), requires_grad = True)
        self.target_prototypes_ema = Variable(torch.zeros([self.k, 256]), requires_grad = True)
        # Optimizer
        #if self.cfg.opt.kind == "SGD":
        #    self.optimizer = torch.optim.SGD(
        #        params, momentum=self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay)
        
        self.queue = torch.zeros(cfg.queue_length, 256).to(self.device)
        self.queue_label = torch.zeros(cfg.queue_length, 1).to(self.device)
        self.queue_out = torch.zeros(cfg.queue_length, cfg.data.num_classes).to(self.device)
        self.use_the_queue = False
        self.use_the_queue_out = False
        self.bs=cfg.bs
        self.fbs=cfg.fbs
        objective_vectors = torch.load(os.path.join(cfg.root,'pretrained/gta2citylabv2', 'prototypes_on_{}_from_{}.pth'.format(cfg.data.target.dataset, cfg.model.backbone)))
        #objective_vectors = torch.load(os.path.join('pretrained_models/gta2citylabv2', 'prototypes_on_{}_from_{}.pth'.format(opt.tgt_dataset, opt.model_name)))
        lisaa=[0,1,2,3,4,6,8,9,10,13,15,17,18]
        objective_vectors=objective_vectors[lisaa]
        #print("objective_vectors",objective_vectors.shape)
        objective_vectors = torch.Tensor(objective_vectors).to(0)

        # init prototype layer
        with torch.no_grad():
            w = torch.nn.functional.normalize(objective_vectors, dim=1, p=2)
            self.prototypes.data = w
            

        #param

        #sinkhorn
        self.temperature=cfg.temperature
        self.sinkhorn_iterations=cfg.sinkhorn_iterations
        self.epsilon=cfg.epsilon
        self.class_distribution = torch.zeros([self.num_classes])
        optimizer_params = {'lr':cfg.opt.lr, 'weight_decay':self.cfg.opt.weight_decay, 'momentum':self.cfg.opt.momentum}
        if self.cfg.opt.kind == "SGD":
            optimizer_cls = torch.optim.SGD
            self.optimizer = optimizer_cls([{'params':self.model.module.get_1x_lr_params_NOscale(), 'lr':optimizer_params['lr']}, 
                                           {'params':self.model.module.get_10x_lr_params(), 'lr':optimizer_params['lr']*10},
                                           {'params':self.prototypes, 'lr':optimizer_params['lr']*5},
                                           {'params':self.target_prototypes, 'lr':optimizer_params['lr']*5}], **optimizer_params)
        elif self.cfg.opt.kind == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(
                0.9, 0.99), weight_decay=self.cfg.opt.weight_decay)
        else:
            raise NotImplementedError()
        self.lr_factor = 10


        # Memory
        self.MQ_size=cfg.MQ_size
        self.n_batch = int(cfg.MQ_size/cfg.fbs)    
        self.memqueue = MemoryQueue(256, cfg.fbs, self.n_batch, 0.1).cuda()
        self.gamma=cfg.gamma
        self.mu=cfg.mu
        self.temp=cfg.temp
        self.lam=cfg.lam
        self.beta = None
        
        # Source
        if self.cfg.data.source.dataset == 'synthia':
            source_train_dataset = SYNTHIA_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = SYNTHIA_Dataset(split='val', **self.cfg.data.source.kwargs)
        elif self.cfg.data.source.dataset == 'gta5':
            source_train_dataset = GTA5_Dataset(split='train', **self.cfg.data.source.kwargs)
            source_val_dataset = GTA5_Dataset(split='val', **self.cfg.data.source.kwargs)
        else:
            raise NotImplementedError()
        self.source_dataloader = DataLoader(
            source_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.source_val_dataloader = DataLoader(
            source_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Target
        if self.cfg.data.target.dataset == 'cityscapes':
            target_train_dataset = City_Dataset(split='train', **self.cfg.data.target.kwargs)
            target_val_dataset = City_Dataset(split='val', **self.cfg.data.target.kwargs)
        else:
            raise NotImplementedError()
        self.target_dataloader = DataLoader(
            target_train_dataset, shuffle=True, drop_last=True, **self.cfg.data.loader.kwargs)
        self.target_val_dataloader = DataLoader(
            target_val_dataset, shuffle=False, drop_last=False, **self.cfg.data.loader.kwargs)

        # Perturbations
        if self.cfg.lam_aug > 0:
            self.aug = get_augmentation()

    def train(self):

        # Loop over epochs
        self.continue_training = True
        while self.continue_training:

            # Train for a single epoch
            self.train_one_epoch()

            # Use EMA params to evaluate performance
            self.ema.apply_shadow()
            self.ema.model.eval()
            self.ema.model.cuda()

            # Validate on source (if possible) and target
            #if self.cfg.data.source_val_iterations > 0:
            #    self.validate(mode='source')
            PA, MPA, MIoU, FWIoU = self.validate()
            
            self.evaluator.Print_Every_class_Eval(
                out_16_13=(int(self.num_classes) in [16, 13]))
            # Restore current (non-EMA) params for training
            self.ema.restore()

            # Log val results
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

            # Save checkpoint if new best model
            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.iter
                self.logger.info("=> Saving a new best checkpoint...")
                self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                    self.best_MIou, self.best_iter))
                self.save_checkpoint('best.pth')
            else:
                self.logger.info("=> The MIoU of val did not improve.")
                self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                    self.best_MIou, self.best_iter))
            self.epoch += 1

        # Save final checkpoint
        self.logger.info("=> The best MIou was {:.3f} at iter {}".format(
            self.best_MIou, self.best_iter))
        self.logger.info(
            "=> Saving the final checkpoint to {}".format('final.pth'))
        self.save_checkpoint('final.pth')

    def train_one_epoch(self):

        # Load and reset
        self.model.train()
        self.evaluator.reset()
        
        # Helper
        def unpack(x):
            return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

        # Training loop
        total = min(len(self.source_dataloader), len(self.target_dataloader))
        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(
            zip(self.source_dataloader, self.target_dataloader),
            total=total, desc=f"Epoch {self.epoch + 1}",mininterval=60
        )):

            # Learning rate
            self.poly_lr_scheduler(optimizer=self.optimizer)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.iter)
            
            # Losses
            losses = {}

            ##########################
            # Source supervised loss #
            ##########################
            x, y, _ = batch_s

            if True:  # For VS Code collapsing
                with torch.autograd.set_detect_anomaly(True):
                    # Data
                    x = x.to(self.device)
                    y = y.squeeze(dim=1).to(device=self.device,
                                            dtype=torch.long, non_blocking=True)
                
                
                    # Forward
                    s_pred = self.model(x)
                    source_out=s_pred['out']
                    source_feat=s_pred['feat']
             
                    input_size = x.size()[2:]
                    feat_SL = source_feat.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs,-1,256)
                    
                    feat_SL_DS = torch.nn.functional.normalize(feat_SL)
                    
                    #feat_SL_DS = torch.nn.functional.normalize(feat_SL_DS, dim = 2, p=2)
                    #feat_SL = feat_SL.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs, -1, 256)
                    #feat_SL_DS=feat_SL_DS.permute(0,2,3,1)
                    
                    #out=torch.ones(size=(1,65*65,self.num_classes)).to(self.device)
                    out=torch.ones(size=(self.bs,91*161,self.num_classes)).to(self.device)
                    #source Supervision
                     
                    for i in range(feat_SL_DS.shape[0]):
                        proto = torch.nn.functional.normalize(self.prototypes, dim = 1, p=2)
                        proto=proto.to(self.device)
                        outs = torch.mm(feat_SL_DS[i], proto.t())
                        out[i]=outs
                    

                    """
                    for i in range(feat_SL_DS.shape[1]):
                        for j in range(feat_SL_DS.shape[2]):
                                    
                            proto = torch.nn.functional.normalize(self.prototypes, dim = 1, p=2)
                            proto=proto.to(self.device)
                            print("proto",proto.shape)
                            print("proto",proto)
                            print("feat_SL_DS[0][i][j]",feat_SL_DS[0][i][j].shape)
                            print("feat_SL_DS[0][i][j]",feat_SL_DS[0][i][j])
                            out_feature = torch.mm(feat_SL_DS[0][i][j].unsqueeze(0), proto.t())
                                            
                            print("out",out.shape)
                            print("out",out)
                            exit()
                            out[0][i][j]=out_feature
                    """
                    #out=out.permute(0,3,1,2)
 
                    out=out.transpose(1,2).contiguous().view(self.bs,self.num_classes,91,161)
           
                    out=F.interpolate(out,size=input_size,mode='bilinear', align_corners=True)
                    
                    #y=F.interpolate(y,size=input_size,mode='bilinear', align_corners=True)
                    #print(out)
                    #print(out.shape)
                    loss_source=self.loss(out,y)
                    #print(loss_source)
                    losses['source_main'] = loss_source.cpu().item()
                    loss_source.backward(retain_graph=True)
                    
                    del x, y
            ######################
            # Target Pseudolabel #
            ######################
            
            x, _, _ = batch_t
            x = x.to(self.device)

        
            target_pred = self.model(x.to(self.device))
                
            target_pred = self.model(x)
            target_out=target_pred['out']
            target_feat=target_pred['feat']

            feat_SL = target_pred['feat']
            out_SL = target_pred['out']
            
            #  ============ self-labeling loss ... ============  
            feat_SL = feat_SL.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs, -1, 256)
            # randomly sampling pixel features
            
            feat_SL2=feat_SL[i].unsqueeze(0)
            rand_index = torch.randperm(feat_SL2.shape[1])
            feat_SL2 = feat_SL2[:,rand_index]
        
            feat_SL_DS = feat_SL2[:, :self.fbs]
            feat_SL_DS = torch.nn.functional.normalize(feat_SL_DS, dim = 2, p=2)
        
        
            
            proto = torch.nn.functional.normalize(self.prototypes, dim = 1, p=2)
            proto=proto.to(self.device)
            target_proto = torch.nn.functional.normalize(self.target_prototypes, dim = 1, p=2)
            target_proto=target_proto.to(self.device)
        
            feat_SL_DS2=feat_SL_DS.clone().squeeze(0)
            #feat_SL_DS3=feat_SL_DS3.clone().squeeze(0)

            after_lincls_t = torch.mm(feat_SL_DS2, proto.t())
            after_cluhead_t = torch.mm(feat_SL_DS2, target_proto.t())
            loss_CCD_total=0
            loss_all_total=0 
            loss_PCD_total=0
            
            if self.iter > 100:
                #source_prototype = classifier.module.ProtoCLS.fc.weight
                
                if self.beta is None:
                    self.beta = ot.unif(proto.size()[0])

                # fill input features with memory queue
                fill_size_uot = self.MQ_size
                mqfill_feat_t = self.memqueue.random_sample(fill_size_uot)
                ubot_feature_t = torch.cat([mqfill_feat_t, feat_SL_DS2], 0)
                full_size = ubot_feature_t.size(0)
                # Adaptive filling
                newsim, fake_size = adaptive_filling(ubot_feature_t, proto, self.gamma, self.beta, fill_size_uot)
            
                
                # UOT-based CCD
                high_conf_label_id, high_conf_label, pred_label, new_beta = ubot_CCD(newsim, self.beta, fake_size=fake_size, 
                                                                            fill_size=fill_size_uot, mode='minibatch')
                # adaptive update for marginal probability vector
                self.beta = self.mu*self.beta + (1-self.mu)*new_beta

                # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
                # Due to mini-batch sampling, current mini-batch samples might be all target-private. 
                # (especially when target-private samples dominate target domain, e.g. OfficeHome)
                if high_conf_label_id.size(0) > 0:
                    criterion = nn.CrossEntropyLoss().cuda()
                    loss_CCD = criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
                    loss_CCD_total+=loss_CCD
                    del loss_CCD
                else:
                    loss_CCD_total += 0.0
            else:
                loss_CCD_total += 0.0
            self.id_target=[]
            for i in range(self.iter*256,self.iter*256+256):
                self.id_target.append(i)
            self.id_target=torch.tensor(self.id_target)

            self.memqueue.update_queue(feat_SL_DS2, self.id_target.cuda())
            #self.prototypes=weight_norm(self.prototypes.clone())
                
            minibatch_size = feat_SL_DS2.size(0)
            
            # obtain nearest neighbor from memory queue and current mini-batch
            feat_mat2 = torch.matmul(feat_SL_DS2, feat_SL_DS2.t()) / self.temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().cuda()
            feat_mat2.masked_fill_(mask, -1/self.temp)
            

            nb_value_tt, nb_feat_tt = self.memqueue.get_nearest_neighbor(feat_SL_DS2, self.id_target.cuda())
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
            values, indices = torch.max(neighbor_candidate_sim, 1)
            neighbor_norm_feat = torch.zeros((minibatch_size, feat_SL_DS2.shape[1])).cuda()
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), feat_SL_DS2], 0)
                neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]
                
            #neighbor_output = cluster_head(neighbor_norm_feat) #nn의 output
            neighbor_output=torch.mm(neighbor_norm_feat, target_proto.t())
        
            # fill input features with memory queue
            fill_size_ot = self.k
            
            mqfill_feat_t = self.memqueue.random_sample(fill_size_ot)
            
            mqfill_output_t = torch.mm(mqfill_feat_t, target_proto.t())
            

            # OT process
            # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
            
            S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
        
            S_tt *= self.temp
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]

            # compute loss_PCD
            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:]))
                sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            loss_PCD = (loss_global + loss_local) / 2
            loss_PCD_total+=loss_PCD
            #loss_all.backward()
            #losses['loss_PCD'] = loss_PCD.cpu().item()
            #del loss_PCD
            self.memqueue.update_queue(feat_SL_DS2, self.id_target.cuda())
            #self.prototypes=weight_norm(self.prototypes) # very important for proto-classifier
            #self.target_prototypes=weight_norm(self.target_prototypes)
            if loss_CCD_total!=0:
                loss_CCD_total.backward(retain_graph=True)
                losses['loss_CCD'] = loss_CCD_total.cpu().item()
            else:
                losses['loss_CCD']=0.0
            loss_PCD_total.backward()
            losses['loss_PCD'] = loss_PCD_total.cpu().item()
            
            
     
            if self.epoch>=0 and batch_idx > 0 and batch_idx % 500 == 0:

                # Use EMA params to evaluate performance
                self.ema.apply_shadow()
                self.ema.model.eval()
                self.ema.model.cuda()

                # Validate on source (if possible) and target
                # if self.cfg.data.source_val_iterations > 0:
                #     self.validate(mode='source')
                PA, MPA, MIoU, FWIoU = self.validate()

                # Restore current (non-EMA) params for training
                self.ema.restore()

                # Log val results
                self.writer.add_scalar('PA', PA, self.epoch)
                self.writer.add_scalar('MPA', MPA, self.epoch)
                self.writer.add_scalar('MIoU', MIoU, self.epoch)
                self.writer.add_scalar('FWIoU', FWIoU, self.epoch)

                # Save checkpoint if new best model
                self.current_MIoU = MIoU
                is_best = MIoU > self.best_MIou
                if is_best:
                    self.best_MIou = MIoU
                    self.best_iter = self.iter
                    self.logger.info("=> Saving a new best checkpoint...")
                    self.logger.info("=> The best val MIoU is now {:.3f} from iter {}".format(
                        self.best_MIou, self.best_iter))
                    self.save_checkpoint('best.pth')
                else:
                    self.logger.info("=> The MIoU of val did not improve.")
                    self.logger.info("=> The best val MIoU is still {:.3f} from iter {}".format(
                        self.best_MIou, self.best_iter))




            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update model EMA parameters each step
            self.ema.update_params()
            # weight_norm
            
             # very important for proto-classifier
            #self.target_prototypes=weight_norm(self.target_prototypes)
            # Calculate total loss
            
            total_loss = sum(losses.values())

            # Log main losses
            for name, loss in losses.items():
                self.writer.add_scalar(f'train/{name}', loss, self.iter)

            # Log
            if batch_idx % 100 == 0:
                print(self.prototypes.data)
                log_string = f"[Epoch {self.epoch}]\t"
                log_string += '\t'.join([f'{n}: {l:.3f}' for n, l in losses.items()])
                self.logger.info(log_string)

            # Increment global iteration counter
            self.iter += 1

            # End training after finishing iterations
            if self.iter > self.cfg.opt.iterations:
                self.continue_training = False
                return

        # After each epoch, update model EMA buffers (i.e. batch norm stats)
        self.ema.update_buffer()
 
    @torch.no_grad()
    def sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
    #     dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            self.class_distribution=self.class_distribution.to(self.device)
            Q *= self.class_distribution.unsqueeze(1)

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        # Q = torch.argmax(Q, 0)
        return Q.t()    
    
    @ torch.no_grad()
    def validate(self, mode='target'):
        """Validate on target"""
        self.logger.info('Validating')
        self.evaluator.reset()
        self.model.eval()

        # Select dataloader
        if mode == 'target':
            val_loader = self.target_val_dataloader
        elif mode == 'source':
            val_loader = self.source_val_dataloader
        else:
            raise NotImplementedError()

        # Loop
        for val_idx, (x, y, id) in enumerate(tqdm(val_loader, desc=f"Val Epoch {self.epoch + 1}")):
            if mode == 'source' and val_idx >= self.cfg.data.source_val_iterations:
                break

            # Forward
            x = x.to(self.device)
            y = y.to(device=self.device, dtype=torch.long)
            pred = self.model(x)
            #pred=pred['output']
            
            source_feat=pred['feat']
            source_out=pred['out']
            input_size = x.size()[2:]

            feat_SL = source_feat.transpose(1, 2).transpose(2, 3).contiguous().view(self.bs,-1,256)
            
            feat_SL_DS = torch.nn.functional.normalize(feat_SL)
            out=torch.ones(size=(self.bs,65*65,self.num_classes)).to(self.device)
            #source Supervision
                        
            stopThr = 1e-6
            proto = torch.nn.functional.normalize(self.prototypes, dim = 1, p=2)
            proto=proto.to(self.device)
            feat_SL_DS2=feat_SL_DS.clone().squeeze(0)
            newsim, fake_size = adaptive_filling(feat_SL_DS2, proto, self.gamma, self.beta, 0,stopThr=stopThr)
            high_conf_label_id, high_conf_label, pred_label, new_beta = ubot_CCD(newsim, self.beta, fake_size=fake_size, 
                                                                            fill_size=0, mode='minibatch',stopThr=stopThr)
            
            pred_label=pred_label.unsqueeze(0)
            
            out=pred_label.contiguous().view(self.bs,65,65).unsqueeze(0)
            
            out=out.type(torch.FloatTensor)
            
            pred=F.interpolate(out,size=input_size,mode='bilinear', align_corners=True)

           
            # Convert to numpy
            label = y.squeeze(dim=1).cpu().numpy()
            
            argpred=pred.squeeze(0)
            argpred=argpred.type(torch.int64).cpu().numpy()
            self.evaluator.add_batch(label, argpred)
       
        # Tensorboard images
        vis_imgs = 2
        images_inv = inv_preprocess(x.clone().cpu(), vis_imgs, numpy_transform=True)
        labels_colors = decode_labels(label, vis_imgs)
        preds_colors = decode_labels(argpred, vis_imgs)
        for index, (img, lab, predc) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image(str(index) + '/images', img, self.epoch)
            self.writer.add_image(str(index) + '/labels', lab, self.epoch)
            self.writer.add_image(str(index) + '/preds', predc, self.epoch)

        # Calculate and log
        if self.cfg.data.source.kwargs.class_16:
            PA = self.evaluator.Pixel_Accuracy()
            MPA_16, MPA_13 = self.evaluator.Mean_Pixel_Accuracy()
            MIoU_16, MIoU_13 = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU_13 = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC_16, PC_13 = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(
                self.epoch, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
            self.logger.info('Epoch:{:.3f}, PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(
                self.epoch, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA_16', MPA_16, self.epoch)
            self.writer.add_scalar('MIoU_16', MIoU_16, self.epoch)
            self.writer.add_scalar('FWIoU_16', FWIoU_16, self.epoch)
            self.writer.add_scalar('MPA_13', MPA_13, self.epoch)
            self.writer.add_scalar('MIoU_13', MIoU_13, self.epoch)
            self.writer.add_scalar('FWIoU_13', FWIoU_13, self.epoch)
            PA, MPA, MIoU, FWIoU = PA, MPA_13, MIoU_13, FWIoU_13
        else:
            PA = self.evaluator.Pixel_Accuracy()
            MPA = self.evaluator.Mean_Pixel_Accuracy()
            MIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            PC = self.evaluator.Mean_Precision()
            self.logger.info('Epoch:{:.3f}, PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(
                self.epoch, PA, MPA, MIoU, FWIoU, PC))
            self.writer.add_scalar('PA', PA, self.epoch)
            self.writer.add_scalar('MPA', MPA, self.epoch)
            self.writer.add_scalar('MIoU', MIoU, self.epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.epoch)
        self.evaluator.Print_Every_class_Eval(out_16_13=(int(self.num_classes) in [16, 13]))
        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename='checkpoint.pth'):
        torch.save({
            'epoch': self.epoch + 1,
            'iter': self.iter,
            'state_dict': self.ema.model.state_dict(),
            'shadow': self.ema.shadow,
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')

        # Get model state dict
        if not self.cfg.train and 'shadow' in checkpoint:
            state_dict = checkpoint['shadow']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove DP/DDP if it exists
        state_dict = {k.replace('module.', ''): v for k,
                      v in state_dict.items()}

        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.logger.info(f"Model loaded successfully from {filename}")

        # Load optimizer and epoch
        if self.cfg.train and self.cfg.model.resume_from_checkpoint:
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f"Optimizer loaded successfully from {filename}")
            if 'epoch' in checkpoint and 'iter' in checkpoint:
                self.epoch = checkpoint['epoch']
                self.iter = checkpoint['iter'] if 'iter' in checkpoint else checkpoint['iteration']
                self.logger.info(f"Resuming training from epoch {self.epoch} iter {self.iter}")
        else:
            self.logger.info(f"Did not resume optimizer")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
        init_lr = self.cfg.opt.lr if init_lr is None else init_lr
        iter = self.iter if iter is None else iter
        max_iter = self.cfg.opt.iterations if max_iter is None else max_iter
        power = self.cfg.opt.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr


@hydra.main(config_path='configs', config_name='gta5')
def main(cfg: DictConfig):

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Monitoring
    if cfg.wandb:
        import wandb
        wandb.init(project='pixmatch', name=cfg.name, config=cfg, sync_tensorboard=True)
    writer = SummaryWriter(cfg.name)

    # Trainer
    trainer = Trainer(cfg=cfg, logger=logger, writer=writer)

    # Load pretrained checkpoint
    if cfg.model.checkpoint:
        assert Path(cfg.model.checkpoint).is_file(), f'not a file: {cfg.model.checkpoint}'
        trainer.load_checkpoint(cfg.model.checkpoint)

    # Print configuration
    logger.info('\n' + OmegaConf.to_yaml(cfg))

    # Train
    if cfg.train:
        trainer.train()

    # Evaluate
    else:
        trainer.validate()
        trainer.evaluator.Print_Every_class_Eval(
            out_16_13=(int(cfg.data.num_classes) in [16, 13]))


if __name__ == '__main__':
    main()
