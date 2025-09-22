"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import absolute_import

import time
import datetime
import numpy as np
import os.path as osp

import torch
from torch import nn

from utils.arguments import get_args, set_log, print_args, set_gpu
from utils.util import set_random_seed
from dataset import dataset_manager, PRCC,LTCC
from dataset.LTCC import cap_gen,cap_gen_attr
from data_process import dataset_loader_cc, dataset_loader
from losses.triplet_loss import TripletLoss,TripletLoss_hard
from losses.con_loss import con_loss
from losses.CE_loss import *
from losses.cross_entropy_loss import CrossEntropyLabelSmooth,CrossEntropyLoss
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from utils.util import load_checkpoint, save_checkpoint
from model import fire,base_block
from model.base_block import *
import train_fire, test_cc, test
from clipS import clip
from clipS.model import *
from torch import optim
from scheduler.scheduler_factory import create_scheduler
import collections
from collections import defaultdict
from tqdm import tqdm, trange
from collections import Counter
from dataset.LTCC import reassignAttrlabel
from processor_clipreid_stage1 import do_train_stage1


# set_random_seed(605, True)
# @torch.no_grad()
# def extract_features(model, data_loader):
#     # test
#     model.eval() 
#     # metric_logger = utils.MetricLogger(delimiter="  ")
#     # header = 'Evaluation:'
#     print('Computing features for evaluation...')
#     # start_time = time.time()

#     # texts = data_loader.dataset.text   
#     # num_text = len(texts)
#     # text_bs = 256
#     # text_ids = []
#     text_embeds = []
#     # text_atts = []
#     ids = []
#     # for i in range(0, num_text, text_bs):
#     #     text = texts[i: min(num_text, i+text_bs)]
#     #     text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=60, return_tensors="pt").to(device)
#     #     text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
#     #     text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
#     #     text_embeds.append(text_embed)   
        
#     # text_embeds = torch.cat(text_embeds,dim=0)
    
#     image_feats = []
#     image_embeds = []
    
#     dataid_dict = defaultdict(list)
#     feat_dict = defaultdict(list)
#     word_dict= defaultdict(list)
#     # dataid_dict = defaultdict(list)
#     datapid_dict = defaultdict(list)
    
#     # 
    
#     pids = []
#     attrlabels =[]
#     # clothes_ids=[]
#     # des_cloes=[]
#     dataid = 0
#     for imgs,pid,atlb,_, attrlabel,description,des,des_cloth,_ in data_loader:
#         ids.append(atlb)#clothes_id
#         pids.append(pid)
#         attrlabels.append(attrlabel)
#         # clothes_ids.append(clothes_id)
#         # des_cloes.append(des_cloth)
#         imgs = imgs.cuda()#.unsqueeze(0)
#         des,des_cloth = des.cuda(),des_cloth.cuda()#.unsqueeze(0)
#         image_feat = model.encode_image(imgs).float()
#         image_feat = F.normalize(image_feat[:,0],dim=-1)
        
#         image_embeds.append(image_feat)#embed
#         word_embed= model.encode_text(des).float()#_cloth
#         word_embed =F.normalize(word_embed)
#         text_embeds.append(word_embed)  
        
#         for i in range(imgs.shape[0]):
#             dataid_dict[int(atlb[i])].append(dataid)#,pid,attrlabel,clothes_id
#             datapid_dict[int(pid[i])].append(dataid)
#             dataid += 1
#             feat_dict[int(pid[i])].append(image_feat[i].unsqueeze(0))#clothes_id
#             word_dict[int(pid[i])].append(word_embed[i].unsqueeze(0))#clothes_id
#                 # attr2clo_dict[attrlabel[i]].append(int(clothes_id[i]))
#     image_embeds = torch.cat(image_embeds,dim=0)
#     text_embeds = torch.cat(text_embeds,dim=0)
#     return image_embeds,  ids,pids,dataid_dict,feat_dict,datapid_dict,attrlabels,text_embeds,word_dict
# set_random_seed(605, True)
def main():#train(rank,world_size):
    # dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)#
    # torch.cuda.set_device(rank)
    
    args = get_args()
    set_log(args)
    print_args(args)
    use_gpu = set_gpu(args)
    # set_random_seed(args.seed, use_gpu)
    set_random_seed(args.seed, True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/backup/lx/pretrained/',pnum=30)#ViT-L/14
    # ViT_model_m, ViT_preprocess_m = clip.load("ViT-L/14", device=device,download_root='/media/backup/lx/pretrained/',pnum=40) #ViT-L/14
    # ViT_model_m = ViT_model

    print("Initializing dataset {}".format(args.dataset))
    if args.dataset == 'prcc':
        dataset = PRCC.PRCC(dataset_root=args.dataset_root, dataset_filename=args.dataset_filename)
        train_loader, query_sc_loader, query_cc_loader, gallery_loader,train_loader_norm = dataset_loader_cc.get_prcc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    elif args.dataset in ['ltcc', 'deepchange', 'last','vc-clothes','celeb-light']:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader,train_loader_norm = \
            dataset_loader_cc.get_cc_dataset_loader(dataset, args=args, use_gpu=use_gpu)#_nosam
    else:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader.get_dataset_loader(dataset, args=args, use_gpu=use_gpu)

    num_classes = dataset.num_train_pids
    # labels = dataset.attrlabel[:,:30]
    # sample_weight = labels.mean(0)
    
    # model = fire.FIRe(pool_type='maxavg', last_stride=1, pretrain=True, num_classes=num_classes)
    sub_model = TransformerClassifier(ViT_model,num_classes=num_classes)
    model = mainmodel(sub_model, ViT_model, num_classes)#,ViT_model_m
    # classifier = fire.Classifier(feature_dim=model.feature_dim, num_classes=num_classes)

    # class_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=use_gpu)
    class_criterion = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=False)
    metric_criterion = TripletLoss(margin=args.margin)
    # criterion_att = CEL_Sigmoid(sample_weight, attr_idx=30)
    
    ##################################################
    # ViT_model.visual.transformer.VorT=False
    # image_feature, ids,pids,dataid_dict,feat_dict,datapid_dict,attrlabels,text_feature,word_dict= extract_features(model=ViT_model, data_loader=train_loader)#,word_dict,
    # labels = np.concatenate(ids) #cluster.fit_predict(distance2) #
    # pids = np.concatenate(pids)
    # attrlabels = np.concatenate(attrlabels)
    
    # print(labels, type(labels), labels.shape)
    # num_cids = len(feat_dict.keys())
    
    # with torch.no_grad():
    #     for cid in trange(num_cids):
    #         img_feats = torch.concat(feat_dict[cid], dim=0)
    #         txt_feats = torch.concat(word_dict[cid], dim=0)
        
    #         image_centers = img_feats.mean(0)
    #         index = torch.sum(img_feats*image_centers,dim=1)#<0.84
    #         # index = torch.sum((img_feats*txt_feats),dim=1)#ViT_model.logit_scale.exp()*(
    #         sorted_index = sorted(index, reverse=True) 
    #         num_elements = int(len(sorted_index) * 0.98)
    #         top_90_percent = sorted_index[:num_elements] 
    #         th = top_90_percent[-1]
    #         index = index < th  
    #         index = index.cpu().numpy()
    #         a=labels[datapid_dict[cid]]
    #         a[index]=-1
    #         labels[datapid_dict[cid]]=a  

    # label_dict = defaultdict(list)
    # coun = 0
    # for i, la in enumerate(set(labels)):
    #     if la==-1:continue
    #     label_dict[la]=coun
    #     coun+=1
    # for i, la in enumerate(labels):
    #     if la==-1:continue
    #     labels[i]= label_dict[la]
    
    # pseudo_dataset = []
    # for i, (img, pid, clothid, camid,attrlabel,des,des_inv,des_cloth,mask) in enumerate(dataset.train):
    #     if labels[i]==-1:continue
    #     # des_rf = cap_gen_attr(attrlabels[i])
    #     pseudo_dataset.append((img, pid,labels[i] , camid,attrlabel,des,des_inv,des_cloth,mask))#clothid,des_clo,des_rf
    # print(len(pseudo_dataset))
    # # num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # # print(num_clusters)
    # ViT_model.visual.transformer.VorT=True
    # del image_feature, labels,dataid_dict,datapid_dict,feat_dict,text_feature,word_dict
    # torch.cuda.empty_cache()
    
    # dataset.train =pseudo_dataset
    # train_loader, query_loader, gallery_loader = \
    #         dataset_loader_cc.get_cc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    
    clip_params=[]
    for name, param in ViT_model.named_parameters():
        if any(keyword in name for keyword in args.clip_update_parameters):
            print(name, param.requires_grad)
            clip_params.append(param)
        else:
            param.requires_grad = False
            
    # stage1_params=[]
    # for name, param in model.named_parameters():
    #     if "prompt_text_deep" in name :# or "prompt_learner" in name
    #         param.requires_grad = True
    #         print(name, param.requires_grad)
    #         stage1_params.append(param)
        
    
    # clip_params_m=[]
    # for name, param in ViT_model_m.named_parameters():
    #     if any(keyword in name for keyword in args.clip_update_parameters):
    #         print(name, param.requires_grad)
    #         clip_params_m.append(param)
    #     else:
    #         param.requires_grad = False

    lr = args.lr
    epoch_num = args.max_epoch
    # optimizer_stage1 = optim.AdamW([{'params':params} for params in stage1_params],lr=lr, weight_decay=args.weight_decay)
    # scheduler_stage1 = create_scheduler(optimizer_stage1, num_epochs=epoch_num, lr=lr, warmup_t=10)#args.st1_lr,args.stage1_epoch
    
    optimizer = optim.AdamW([{'params':params} for params in clip_params]+[{'params':sub_model.parameters()}],lr=lr, weight_decay=args.weight_decay)#[{'params':params} for params in clip_params_m]+
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=10)

    start_epoch = args.start_epoch  # 0 by default
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        clip_pretrain_dict=checkpoint['clip_model']
       
        ViT_model=build_model(clip_pretrain_dict).cuda()
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if use_gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1  # start from the next epoch

    # scheduler = WarmupMultiStepLR(optimizer, milestones=args.step_milestones, gamma=args.gamma,
    #                               warmup_factor=args.warm_up_factor, last_epoch=start_epoch - 1)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if use_gpu:
        model = nn.DataParallel(model).cuda()
       
    # model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    
    # only test
    if args.evaluate:
        print("Evaluate only")
        if args.dataset == 'prcc':
            test_cc.test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        elif args.dataset == 'ltcc':
            test_cc.test_for_ltcc(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        elif args.dataset in ['deepchange', 'last']:
            test_cc.test_for_cc(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        else:
            test.test(args, query_loader, gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=None)
        return 0
    ###########stage1
    # do_train_stage1(
    #     args,
    #     model,
    #     train_loader_stage1,
    #     optimizer_stage1,
    #     scheduler_stage1
    # )#,args.local_rank
    # train
    torch.cuda.empty_cache()
    print("==> Start training")
    start_time = time.time()
    train_time = 0
    best_mAP, best_rank1 = -np.inf, -np.inf
    best_epoch_mAP, best_epoch_rank1 = 0, 0

    flag = False
    best_mAP_2, best_rank1_2 = -np.inf, -np.inf
    best_epoch_mAP_2, best_epoch_rank1_2 = 0, 0
    isflag=True
    isflag_g=True
    loss_clo=100.
    loss_clo_g=100.
    lamd=0.1
    gama=0.04
    beta=[0.5,0.1]
    # beta=[0.5,0.2,0.1]
    # alpha=[0.5,0.1,0.02]
    k,s=0,0
    # ViT_model.transformer.prompt_text_deep.requires_grad=False
    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        htriloss1,htriloss3 = train_fire.train(args, epoch + 1, dataset, train_loader, train_loader_norm,model,
                         optimizer, scheduler, class_criterion, metric_criterion,lamd,gama,use_gpu)#criterion_att,,ViT_model, ViT_model_m,,classifier,, triplet_hard_criterion
        train_time += round(time.time() - start_train_time)
        torch.cuda.empty_cache()
        # if epoch==4:
        #     lamd=0.04
        # if isflag:
        #     if htriloss3 < loss_clo:
        #         loss_clo = htriloss3
        #     else:
        #         lamd=0.04#0.2
        #         isflag=False
        
                            
        # evaluate
        if (epoch + 1) > args.start_eval_epoch and args.eval_epoch > 0 and (epoch + 1) % args.eval_epoch == 0 \
                or (epoch + 1) == args.max_epoch:
            print("==> Test")
            if args.dataset == 'prcc':
                rank1, mAP = test_cc.test_for_prcc(args, query_sc_loader, query_cc_loader,
                                                   gallery_loader, model,ViT_model, use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            elif args.dataset in ['ltcc','vc-clothes']:
                rank1, mAP = test_cc.test_for_ltcc(args, query_loader, gallery_loader, model,
                                                   use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)#,ViT_model
            elif args.dataset in ['deepchange', 'last','celeb-light']:
                rank1, mAP = test_cc.test_for_cc(args, query_loader, gallery_loader, model,ViT_model,
                                                 use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            else:
                rank1, mAP = test.test(args, query_loader, gallery_loader, model,ViT_model,
                                       use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            if isinstance(rank1, list):
                rank1, rank1_2 = rank1
                mAP, mAP_2 = mAP
                flag = True

            is_best_mAP = mAP > best_mAP
            # if isflag and (not is_best_mAP):
            #     lamd=0.04
            #     isflag=False
            is_best_rank1 = rank1 > best_rank1
            if is_best_mAP:
                best_mAP = mAP
                best_epoch_mAP = epoch + 1
            if is_best_rank1:
                best_rank1 = rank1
                best_epoch_rank1 = epoch + 1

            if flag:
                is_best_mAP_2 = mAP_2 > best_mAP_2
                is_best_rank1_2 = rank1_2 > best_rank1_2
                if is_best_mAP_2:
                    best_mAP_2 = mAP_2
                    best_epoch_mAP_2 = epoch + 1
                if is_best_rank1_2:
                    best_rank1_2 = rank1_2
                    best_epoch_rank1_2 = epoch + 1

            if args.save_checkpoint:
                model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
                # classifier_state_dict = classifier.module.state_dict() if use_gpu else classifier.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                save_checkpoint({
                    'model_state_dict': model_state_dict,
                   
                    # 'clip_model': ViT_model.module.state_dict(),
                    # 'clip_model_m': ViT_model_m.module.state_dict(),
                    
                    'optimizer_state_dict': optimizer_state_dict,
                    'rank1': rank1,
                    'mAP': mAP,
                    'epoch': epoch,
                }, is_best_mAP, is_best_rank1, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) +
                                                        '_mAP_' + str(round(mAP * 100, 2)) + '_rank1_' + str(
                    round(rank1 * 100, 2)) + '.pth'))
        # torch.cuda.empty_cache()
    print("==> Best mAP {:.4%}, achieved at epoch {}".format(best_mAP, best_epoch_mAP))
    print("==> Best Rank-1 {:.4%}, achieved at epoch {}".format(best_rank1, best_epoch_rank1))
    if flag:
        print("==> Best mAP_2 {:.4%}, achieved at epoch {}".format(best_mAP_2, best_epoch_mAP_2))
        print("==> Best Rank-1_2 {:.4%}, achieved at epoch {}".format(best_rank1_2, best_epoch_rank1_2))

    # time using info
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    
# def main():
#     world_size = 2  # 使用双卡
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定使用 0 号和 1 号 GPU
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "6666"#12355
#     # 使用 spawn 启动多进程训练
#     mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()