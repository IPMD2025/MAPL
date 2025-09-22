import logging
import os
import torch
import torch.nn as nn
from utils.util import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from losses.supcontrast import SupConLoss

def do_train_stage1(args,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
            ):#  local_rank
    checkpoint_period = args.stage1_epoch
    device = "cuda"
    epochs = args.max_epoch
    log_period = args.log_period 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    # if device:
    #     model.to(local_rank)
    #     if torch.cuda.device_count() > 1:
    #         print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    # image_features = []
    labels = []
    # with torch.no_grad():
    #     for n_iter, (imgs, pid, clothid, camid, attrlabel,des,des_inv,des_cloth) in enumerate(train_loader_stage1):
    #         imgs = imgs.to(device)
    #         target = pid.to(device)
    #         with amp.autocast(enabled=True):
    #             image_feature = model(imgs=imgs, pids=target, get_image = True)
    #             for i, img_feat in zip(target, image_feature):
    #                 labels.append(i)
    #                 image_features.append(img_feat.cpu())
    #     labels_list = torch.stack(labels, dim=0).cuda() #N
    #     image_features_list = torch.stack(image_features, dim=0).cuda()

    #     batch = args.train_batch
    #     num_image = labels_list.shape[0]
    #     i_ter = num_image // batch
    # del labels, image_features
    model.train()
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        
        

        # iter_list = torch.randperm(num_image).to(device)
        for n_iter, (imgs, pid, clothid, camid, attrlabel,des,des_inv,des_cloth) in enumerate(train_loader_stage1):
        # for i in range(i_ter+1):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            target = pid.to(device)
            clothid = clothid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(imgs=imgs, pids=target, get_image = True)
                text_features = model(pids = target, get_text = True,des_cloth=des_cloth)
            # if i != i_ter:
            #     b_list = iter_list[i*batch:(i+1)* batch]
            # else:
            #     b_list = iter_list[i*batch:num_image]
            
            # target = labels_list[b_list]
            # image_features = image_features_list[b_list]
            # with amp.autocast(enabled=True):
            #     text_features = model(pids = target, get_text = True)
            # loss_i2t = xent(image_features, text_features, target, target)
            # loss_t2i = xent(text_features, image_features, target, target)

            # loss = loss_i2t + loss_t2i
            loss = get_contrastive_loss(image_feature,text_features,model.module.ViT_model,clothid)
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), imgs.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))#logger.info

        if epoch % checkpoint_period == 0:
            # if cfg.MODEL.DIST_TRAIN:
            #     if dist.get_rank() == 0:
            #         torch.save(model.state_dict(),
            #                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            # else:
            torch.save(model.state_dict(),
                        os.path.join(args.save_dir, 'vl14' + '_stage1_{}.pth'.format(epoch)))
        scheduler.step(epoch)
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    print("Stage1 running time: {}".format(total_time))#logger.info

def get_contrastive_loss(image_feat, text_feat,model, idx=None):
        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = image_feat#allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = text_feat#allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() 
        logits=logits* model.logit_scale.exp()
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            return (loss_i2t + loss_t2i) / 2
        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = idx#allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
            return (loss_i2t + loss_t2i) / 2