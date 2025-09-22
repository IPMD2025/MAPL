import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.vit import *
from clipS import clip
from clipS.model import *
import numpy as np
# from models.layers import ResidualAttention,TransformerDecoder
# from models.pre_peta_random import petabaseDataset
import copy
from torch.nn import Parameter
from utils.arguments import get_args
import math
from functools import reduce
from operator import mul
args = get_args()

class mainmodel(nn.Module):
    def __init__(self, model,ViT_model,num_classes,get_image = False, get_text = False):#,ViT_model_m
        super().__init__()
        self.model=model
        self.ViT_model=ViT_model
        # self.ViT_model_m=ViT_model_m
        # self.prompt_learner = PromptLearner(num_classes, ViT_model.dtype, ViT_model.token_embedding)
        self.num_classes = num_classes
        
    def forward(self, imgs=None,des=None,des_inv=None,des_cloth=None,pids=None,cloth_ids=None,mask=None,get_image= False,get_text= False):#camid=None,,txtft=None
        return self.model(imgs,self.ViT_model,des,des_inv,des_cloth,pids,cloth_ids,mask,get_image,get_text)#,self.ViT_model_m,camid,self.prompt_learner,txtft,
class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        dtype = torch.float32
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx
        
    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts     
class TransformerClassifier(nn.Module):
    def __init__(self, ViT_model,num_classes,attr_num=35,attr_words='attribute', dim=768, pretrain_path='/media/backup/lx/pretrained/jx_vit_base_p16_224-80ecf9dd.pth',**kwargs):#
        super().__init__()
        self.attr_num = attr_num
        self.dim=dim
        self.dim_c=dim
        # self.word_embed = nn.Linear(dim, dim)
        # self.adapter = Adapter(dim, 4).cuda()
        # self.prompt_learner = PromptLearner(num_classes, ViT_model.dtype, ViT_model.token_embedding)
        attr_words = [
            'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
            
            'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
            'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
            'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
            'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',

            'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
            'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
            'A male pedestrian']

        
        self.lmbd=0
        self.patch=256
        width=1024
        self.head = nn.Linear(self.dim, num_classes,bias=False)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        self.bn_g=nn.BatchNorm1d(self.dim)
        self.bn_g.bias.requires_grad_(False)
        self.bn_g.apply(self._init_kaiming)
        
        
        
        if args.ablation=='no':
            self.bn_a=nn.BatchNorm1d(dim)
            self.bn_a.bias.requires_grad_(False)
            self.bn_a.apply(self._init_kaiming)
            
            self.bn_i=nn.BatchNorm1d(dim)
            self.bn_i.bias.requires_grad_(False)
            self.bn_i.apply(self._init_kaiming)
            
            self.bn_c=nn.BatchNorm1d(dim)
            self.bn_c.bias.requires_grad_(False)
            self.bn_c.apply(self._init_kaiming)
            
            self.bn_f=nn.BatchNorm1d(dim)
            self.bn_f.bias.requires_grad_(False)
            self.bn_f.apply(self._init_kaiming)
        elif args.ablation=='allandinv':
            self.bn_a=nn.BatchNorm1d(dim)
            self.bn_a.bias.requires_grad_(False)
            self.bn_a.apply(self._init_kaiming)
            self.bn_i=nn.BatchNorm1d(dim)
            self.bn_i.bias.requires_grad_(False)
            self.bn_i.apply(self._init_kaiming)
            
            self.bn_f=nn.BatchNorm1d(dim)
            self.bn_f.bias.requires_grad_(False)
            self.bn_f.apply(self._init_kaiming)
        elif args.ablation=='featandall':
            self.bn_f=nn.BatchNorm1d(dim)
            self.bn_f.bias.requires_grad_(False)
            self.bn_f.apply(self._init_kaiming)
        elif args.ablation=='featandclo' or args.ablation=='noattr':
            # self.bn_c=nn.BatchNorm1d(self.dim_c)
            # self.bn_c.bias.requires_grad_(False)
            # self.bn_c.apply(self._init_kaiming)
            
            self.bn_f=nn.BatchNorm1d(self.dim)
            self.bn_f.bias.requires_grad_(False)
            self.bn_f.apply(self._init_kaiming)
            
            # self.bn_mc=nn.BatchNorm1d(self.dim_c)
            # self.bn_mc.bias.requires_grad_(False)
            # self.bn_mc.apply(self._init_kaiming)
            
            
            self.bn_tm=nn.BatchNorm1d(self.dim)
            self.bn_tm.bias.requires_grad_(False)
            self.bn_tm.apply(self._init_kaiming)
            
            self.bn_m=nn.BatchNorm1d(self.dim)
            self.bn_m.bias.requires_grad_(False)
            self.bn_m.apply(self._init_kaiming)
            
            self.head_m = nn.Linear(self.dim, num_classes,bias=False)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
            self.head_m.apply(self._init_weights)
        elif args.ablation=='nomask' or args.ablation=='nomaskandnoattr':
            self.bn_c=nn.BatchNorm1d(self.dim_c)
            self.bn_c.bias.requires_grad_(False)
            self.bn_c.apply(self._init_kaiming)
            
            self.bn_f=nn.BatchNorm1d(self.dim)
            self.bn_f.bias.requires_grad_(False)
            self.bn_f.apply(self._init_kaiming)
            
            # self.bn_i=nn.BatchNorm1d(dim)
            # self.bn_i.bias.requires_grad_(False)
            # self.bn_i.apply(self._init_kaiming)
            
            # self.head_m = nn.Linear(dim, num_classes,bias=False)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
            # self.head_m.apply(self._init_weights)
        
        # self.head_f = NormalizedClassifier(dim, 2*num_classes)#nn.Linear(dim, num_classes,bias=False)
        # self.head_f.apply(self._init_classifier)
        
        
        # self.head_adv = nn.Linear(dim, dim)
        # self.head_adv.apply(self._init_weights)

        # vit1=vit_base()
        # vit1.load_param(pretrain_path)
        # self.blocks_t=vit1.blocks[-1:]
        # self.norm_t=vit1.norm
        
        # # self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        # self.bn_p = nn.BatchNorm1d(self.attr_num)#num_classes

        
        # self.text = clip.tokenize(attr_words).cuda()
        # self.bn_des = nn.BatchNorm1d(30)
        # self.cls_part_token=nn.Parameter(torch.zeros(1, self.lmbd, dim))
        # val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + dim))
        # nn.init.uniform_(self.cls_part_token.data, -val,val)
        # self.aggregate = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        # self.norm_all = nn.LayerNorm(self.dim)val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + width))
        # val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + width))
        # self.prompt_deep=nn.Parameter(torch.zeros(24,10,1,width))
        # nn.init.uniform_(self.prompt_deep.data, -val,val)
        # self.prompt_deep_m=nn.Parameter(torch.zeros(24,10,1,width))
        # nn.init.uniform_(self.prompt_deep_m.data, -val,val)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.001)#02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def _init_classifier(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
                
    def _init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, imgs,ViT_model,des,des_inv,des_cloth,pids,cloth_ids,mask=None,get_image=None,get_text=None,prompt_learner=None):#,ViT_model_m=None,camid=None,txtft=None,
        # self.word_embed = self.word_embed.cuda().float()
        # features,all_x_cls = self.vit(imgs)
        # if camid!=None:
        #     features = ViT_model.encode_image(imgs,camid)#
        # else:
        if get_image == True:
            features = ViT_model.encode_image(imgs,pd=True).float()#features_last,features,features_proj
            #features_last,features,features_proj = features_last.float(),features.float(),features_proj.float()
            return features#features_last[:,0],features[:,0],features_proj[:,0]
        if get_text == True:
            prompts = self.prompt_learner(pids) 
            text_features = ViT_model.encode_text(prompts, self.prompt_learner.tokenized_prompts).float()
            # text_features = ViT_model.encode_text(des_cloth).float()#self.word_embed()
            return text_features
        features = ViT_model.encode_image(imgs).float()#features_last,features,features_proj
        # features_last,features,features_proj = features_last.float(),features.float(),features_proj.float()#self.visual_embed
        B,N,_=features.shape
        if args.ablation=='no':
            destoken = clip.tokenize(des,truncate=True).cuda()
            word_embed_des=ViT_model.encode_text(destoken).cuda().float()
            
            destoken_inv = clip.tokenize(des_inv,truncate=True).cuda()
            word_embed_inv=ViT_model.encode_text(destoken_inv).cuda().float()
            
            # destoken_clo = clip.tokenize(des_cloth,truncate=True).cuda()
            # word_embed_clo=ViT_model.encode_text(destoken_clo).cuda().float()

            feat_ita=torch.cat((self.bn_a(features[:,0]),word_embed_des),dim=1)
            feat_iti=torch.cat((self.bn_i(self.bn_a(features[:,0])),word_embed_inv),dim=1)#self.bn_i()
            feat_itc=torch.cat((self.bn_c(features[:,0]),word_embed_des),dim=1)
            feat_f=self.bn_f(self.bn_i(self.bn_a(features[:,0])))
            feat=self.bn_g(feat_f)
        
        elif args.ablation=='nomaskandnoattr':
            # destoken_clo = clip.tokenize(des_cloth,truncate=True).cuda()
            if self.training:
                # prompts = prompt_learner(pids) 
                # text_features = ViT_model.encode_text(prompts, prompt_learner.tokenized_prompts).float()
                feat_itc = []
                for i in range(30):
                    word_embed_attr=ViT_model.encode_text(des[:,i]).cuda().float()
                    feat_itc.append(torch.cat((self.bn_c(features_proj[:,0]),word_embed_attr),dim=1))#txtft,self.word_embed()
                # for i in range(35,68):
                #     word_embed_attr=ViT_model.encode_text(des[:,i]).cuda().float()
                #     feat_itc.append(torch.cat((self.bn_c(features[:,0]),word_embed_attr),dim=1))
            # word_embed_clo=ViT_model.encode_text(self.text[:30]).float()
            # feat_itc ,_ = ViT_model.forward_aggregate(self.bn_c(features[:,0]), word_embed_clo)
            # feat_itc = self.bn_des(feat_itc)
            # feat_i= self.bn_i(features[:,0])
            feat_f=self.bn_f(features[:,0])#word_embed_clo
            feat=self.bn_g(feat_f)
        
        elif args.ablation=='nomask':
            # destoken_clo = clip.tokenize(des_cloth,truncate=True).cuda()
            word_embed_clo=ViT_model.encode_text(des_cloth).cuda().float()
            feat_itc=torch.cat((self.bn_c(features[:,0]),word_embed_clo),dim=1)#word_embed_clo
            feat_f=self.bn_f(features[:,0])#word_embed_clo
            feat=self.bn_g(feat_f)
        elif args.ablation=='featandclo' or args.ablation=='noattr':
            
            word_embed_clo=ViT_model.encode_text(des_cloth).float()#
            word_embed_inv=ViT_model.encode_text(des_inv).float()#
            if self.training:
                if args.ablation=='featandclo':
                    feat_itc = []
                    word_embed_attr=[]
                    for i in range(35):
                        word_embed_attr.append(ViT_model.encode_text(des[:,i]).cuda().float())
                        feat_itc.append(torch.cat((features[:,0], word_embed_attr[i]),dim=1))#.detach()
                    # for i in range(30,35):
                    #     word_embed_attrinv=ViT_model.encode_text(des[:,i]).cuda().float()
                    #     feat_itc.append(torch.cat((features[:,0],word_embed_attrinv),dim=1))
                    feat_itc.append(torch.cat((features[:,0],word_embed_clo),dim=1))
                    feat_itc.append(torch.cat((features[:,0],word_embed_inv),dim=1))
                else:
                    feat_itc=self.bn_c(features_proj[:,0])
            if mask is not None:
                features_m = ViT_model.encode_image(mask).float()#features_m_last,features_m,features_m_proj
                #features_m_last,features_m,features_m_proj =features_m_last.float(),features_m.float(),features_m_proj.float()
                # feat_im=self.bn_im(features_m[:,0])
                feat_tm=self.bn_tm(features_m[:,0])
                feat_m=self.bn_m(feat_tm)
                # feat_itmc = []
                # for i in range(30):
                #         feat_itmc.append(features_m[:,0])
                # for i in range(30,35):
                #         feat_itmc.append(features_m[:,0])#torch.cat((, word_embed_attr[i]),dim=1)
                # feat_itmc.append(features_m[:,0])#torch.cat((features_m[:,0],word_embed_clo),dim=1)
                # feat_itmc.append(features_m[:,0])
                
                # feat_itmc = []
                # for i in range(30):#[0,1,2,3,4,21,22,23,24]:
                #     feat_itmc.append(self.bn_mc(features_m[:,0]))#_proj,torch.cat((),word_embed_attr_m),dim=1))
                
                # for i in [0,1,2,3,4,21,22,23,24]:
                #     word_embed_attr_m=ViT_model.encode_text(des[:,i]).cuda().float()
                #     feat_itmc.append(torch.cat((self.bn_mc(features_m[:,0]),word_embed_attr_m),dim=1))
                
            # if args.ablation=='featandclo':
            #     # feat_itc=torch.cat((self.bn_c(features[:,0]),word_embed_clo),dim=1)#word_embed_clo
            #     feat_itc ,_ = ViT_model.forward_aggregate(self.bn_c(features[:,0]), word_embed_clo)
            #     feat_itc = self.bn_des(feat_itc)
            # else:
            #     feat_itc=self.bn_c(features[:,0])
            # feat_i = self.bn_i(features[:,0])
            feat_f=self.bn_f(features[:,0])#word_embed_clo
            # feat_i = self.bn_i(feat_f)
            feat=self.bn_g(feat_f)
        elif args.ablation=='noBn':
            destoken_clo = clip.tokenize(des_cloth,truncate=True).cuda()
            word_embed_clo=ViT_model.encode_text(destoken_clo).cuda().float()
            if mask is not None:
                features_m = ViT_model_m.encode_image(mask)#_m
                features_m=(features_m.float())
                feat_tm=features_m[:,0]
                feat_m=self.bn_m(feat_tm)
                feat_itmc=features_m[:,0]   
            feat_itc=torch.cat((features[:,0],word_embed_clo),dim=1)#word_embed_clo
            feat_f=features[:,0]
            feat=self.bn_g(feat_f)
        else:
            # feat_f=self.bn_f(features[:,0])
            feat=self.bn_g(features[:,0])#feat_f
        
        logits_g=self.head(feat)
        logits=[]
        logits.append(logits_g)
        if mask is not None and (args.ablation=='featandclo' or args.ablation=='noBn'):
            logits.append(self.head_m(feat_m))
        # logits=torch.stack(logits,dim=0).cuda()
        
        
        
        
        if self.training:
            if args.ablation =="no":
                return [feat_ita,feat_iti,feat_itc,feat_f],logits#
            elif args.ablation=='nomask' or args.ablation=='nomaskandnoattr':
                return [feat_itc,feat_f,features[:,0]],logits#
            elif args.ablation=='featandall':
                return [feat_ita,feat_f],logits#
            elif args.ablation=='featandclo' or args.ablation=='noattr':
                return [feat_itc,features_m[:,0],feat_tm,feat_f,feat_m,feat,features[:,0]],logits#,features[:,0]
            elif args.ablation=='VLP':
                return [features[:,0]],logits
            elif args.ablation=='abla-ita':
                return [feat_iti,feat_tm,feat_itc,feat_f],logits
            else:
                return [features[:,0]],logits
        else:
            if args.ablation != 'onlyfeat' :
                return feat,logits#,feat_itc#torch.cat((,feat_a),dim=1)
            else:
                return features[:,0],logits

   
        
    

    def get_image_mask(self,N,C):
        # partlist=[[0,(N-1)//2],[0,(N-1)//2],[(N-1)//2,N-1],[(N-1)//2,N-1],[(N-1)//4,(N-1)*3//4],[0,N-1],[0,N-1]]
        P=50
        self.image_mask = torch.zeros(C+P+N, C+P+N)
        self.image_mask[0][C:P+C].fill_(float("-inf"))
        # self.image_mask[1:C,1:C].fill_(float("-inf"))     #8个cls token   
        # self.image_mask[1][C+P+N//2:].fill_(float("-inf"))   #0-hair， 1th，2th，3th块保留  1-age whole attention 2-gender whole attention
        # self.image_mask[2][:C+P+N//2].fill_(float("-inf"))
        # self.image_mask[3][:C+P+N//4].fill_(float("-inf"))   #3-carry 3,4,5,6块保留 [2*2*14+8,6*2*14+8]
        # self.image_mask[3][C+P+N*3//4:].fill_(float("-inf"))  #4-accessory 1,2,3,4,5,6块保留 [6*2*14+8]  
 
        # for i in range(C): 
        #     if i!=0:
        #         self.image_mask[i][C:P+C].fill_(float("-inf"))#
        #     self.image_mask[i][0].fill_(0)#
        #     self.image_mask[i][i].fill_(0)

    def get_groupvice(self,grouporder):
        length=len(grouporder)
        group_vice=[]
        for i in range(length):
            for j in range(length):
                if i==grouporder[j]:
                    group_vice.append(j)
        return group_vice


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
        
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Part_CAM(nn.Module):
    def __init__(self,lmbd):
        super(Part_CAM, self).__init__()
        self.lmbd=lmbd
        

    def forward(self, x,features):
        length = len(x)
        C=1+self.lmbd
        P=50
        
        # b=x[0].shape[0]
        att_tt=[]
        N=features.shape[1]
        feat_cam=[]
        partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
        for d in range(length):
           
            att_tk=x[d][:,0]
            att_pt=x[d][:,1:1+P]
            att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
            att_tk2=att_t1[:,:,0]
            att_pt2=att_t1[:,:,1:1+P]
            att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
            att_tt.append(att)
            # att_tt.append(x[d])
        
        last_map =att_tt[0].float()
        for i in range(1, length):
            last_map = torch.matmul(att_tt[i].float(), last_map)
            
        
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)#1+P
        feat_map=F.relu(features[:,1:1+P])
        feat_cam=last_map1@feat_map
        return feat_cam

# class Part_CAM(nn.Module):
#     def __init__(self,lmbd):
#         super(Part_CAM, self).__init__()
#         self.lmbd=lmbd
        

#     def forward(self, x,features):
#         length = len(x)
#         C=1+self.lmbd
#         P=50
        
#         # b=x[0].shape[0]
#         att_tt=[]
#         N=features.shape[1]
#         feat_cam=[]
#         partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
#         for d in range(length):
           
#             att_tk=x[d][:,0]
#             att_pt=x[d][:,1:1+P]
#             att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
#             att_tk2=att_t1[:,:,0]
#             att_pt2=att_t1[:,:,1:1+P]
#             att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
#             att_tt.append(att)
#             # att_tt.append(x[d])
        
#         last_map =att_tt[0].float()
#         for i in range(1, length):
#             # last_map = torch.matmul(att_tt[i].float(), last_map)
#             last_map = att_tt[i].float()+last_map
#         last_map /=length
        
#         last_map1 = last_map[:,0,1:].unsqueeze(1)#1+P
#         feat_map=F.relu(features[:,:P])
#         feat_cam=last_map1@feat_map
#         return feat_cam
# class Global_CAM(nn.Module):
#     def __init__(self):
#         super(Global_CAM, self).__init__()

#     def forward(self, x,features):
#         length = len(x)
        
#         # feat_cam=[]
#         # N=features.shape[1]
#         # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
#         last_map =x[0].float()
#         for i in range(1, length):
#             last_map = torch.matmul(x[i].float(), last_map)
        
#         last_map1 = last_map[:,0,1:].unsqueeze(1)
#         feat_cam=last_map1@F.relu(features[:,1:])
        
#         return feat_cam  
# class Part_CAM_CLIP_vit_nopatch(nn.Module):
#     def __init__(self,lmbd):
#         super(Part_CAM, self).__init__()
#         self.lmbd=lmbd
        

#     def forward(self, x,features):
#         length = len(x)
#         C=1+self.lmbd
#         P=50
        
#         # b=x[0].shape[0]
#         att_tt=[]
#         N=features.shape[1]
#         feat_cam=[]
#         partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
#         for d in range(length):
            
#             if d==length-1:
#                 att_tk=x[d][:,0]
#                 att_pt=x[d][:,C:C+P]
#                 att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
#                 att_tk2=att_t1[:,:,0]
#                 att_pt2=att_t1[:,:,C:C+P]
#                 att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
#                 att_tt.append(att)
#             else:
#                 att_tk=x[d][:,0]
#                 att_pt=x[d][:,1:1+P]
#                 att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
#                 att_tk2=att_t1[:,:,0]
#                 att_pt2=att_t1[:,:,1:1+P]
#                 att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
#                 att_tt.append(att)
#                 # att_tt.append(x[d])
        
#         last_map =att_tt[0].float()
#         for i in range(1, length):
#             last_map = torch.matmul(att_tt[i].float(), last_map)
        
#         last_map1 = last_map[:,0,1:].unsqueeze(1)#1+P
#         feat_map=F.relu(features[:,C:C+P])
#         feat_cam=last_map1@feat_map
#         return feat_cam

# class Part_CAM_clip_vit(nn.Module):
#     def __init__(self,lmbd):
#         super(Part_CAM, self).__init__()
#         self.lmbd=lmbd
        

#     def forward(self, x,features):
#         length = len(x)
#         C=1+self.lmbd
#         P=50
        
#         b=x[0].shape[0]
#         att_tt=[]
#         N=features.shape[1]
#         feat_cam=[]
#         # partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
#         for d in range(length):
            
#             if d==length-1:
#                 att_tk=x[d][:,0]
#                 att_pt=x[d][:,C:]
#                 att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
#                 att_tk2=att_t1[:,:,0]
#                 att_pt2=att_t1[:,:,C:]
#                 att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
#                 att_tt.append(att)
#             else:
#                 att_tt.append(x[d])
        
#         last_map =att_tt[0].float()
#         for i in range(1, length):
#             last_map = torch.matmul(att_tt[i].float(), last_map)
        
#         last_map1 = last_map[:,0,1:1+P].unsqueeze(1)
#         feat_map=F.relu(features[:,C:C+P])
#         feat_cam=last_map1@feat_map
#         return feat_cam

# class Part_CAM(nn.Module):
#     def __init__(self,lmbd,patch):
#         super(Part_CAM, self).__init__()
#         self.lmbd=lmbd
#         self.patch=patch
        

#     def forward(self, x,features):
#         P=50
#         img_msk_start=[0,0,self.patch // 2,self.patch // 4,0]
#         img_msk_end=[self.patch +1,self.patch // 2,self.patch +1,self.patch * 3 // 4,self.patch +1]
        
#         length = len(x)
#         C=1+self.lmbd
#         b=x[0].shape[0]
#         att_tt=[]
#         N=features.shape[1]
#         feat_cam=[]
#         for d in range(length):
#             if d==length-1:
#                 att_ts=[]
#                 for i in range(C):
#                     att_vit_temp=torch.cat((x[d][:,i].unsqueeze(1),x[d][:,C + P  : ]),dim=1)#+ img_msk_start[i]C + P + img_msk_end[i]
#                     att_vit_noP=torch.cat((att_vit_temp[:,:,i].unsqueeze(-1),att_vit_temp[:,:,C +P : ]),dim=-1)#,+ img_msk_start[i]C + P + img_msk_end[i]
#                 att_tt.append(att_vit_noP)
#             else:
#                 att_temp=torch.cat((x[d][:,0].unsqueeze(1),x[d][:,1+P :]),dim=1)# + P
#                 att_clip_noP=torch.cat((att_temp[:,:,0].unsqueeze(-1),att_temp[:,:,1+P :]),dim=-1)
#                 att_tt.append(att_clip_noP)
        
#         last_map =att_tt[0].float()
#         for i in range(1, length-1):
#             last_map = torch.matmul(att_tt[i].float(), last_map)

#         att_map=[]
#         for f in range(C):
#             last_map_gpcls = torch.matmul(att_tt[length-1][f].float(), last_map)
#             att_map.append(last_map_gpcls)

#         for k in range(C): 
#             last_map1 = att_map[k][:,0,1:].unsqueeze(1)
#             feat_map=F.relu(features[:,C + P  : ])#+ img_msk_start[k]C + P + img_msk_end[k]
#             feat_cam.append(last_map1@feat_map)
#         feat_cam=torch.cat(feat_cam,dim=1)
#         return feat_cam
class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) 

    def forward(self, x):
        w = self.weight  

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)

        

         

