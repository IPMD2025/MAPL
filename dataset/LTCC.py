import glob
import random
import numpy as np
import os.path as osp
import torch
import re
import numpy as np

from dataset.base_image_dataset import BaseImageDataset
attr_words = [
            'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
            
            'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
            'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
            'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
            'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',

            'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
            'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
            'A male pedestrian'#,
            
            # 'A pedestrian in black upper wear', 'A pedestrian in blue upper wear', 'A pedestrian in brown upper wear', 'A pedestrian in green upper wear ', 
            # 'A pedestrian in greyupper wear', 'A pedestrian in a orange upper wear ', 'A pedestrian in pink upper wear ', 'A pedestrian in purple upper wear', 
            # 'A pedestrian in red upper wear', 'A pedestrian in white upper wear', 'A pedestrian in yellow upper wear', 
            
            # 'A pedestrian in black lower wear', 'A pedestrian in blue lower wear', 'A pedestrian in brown lower wear', 'A pedestrian in green lower wear ', 
            # 'A pedestrian in greylower wear', 'A pedestrian in a orange lower wear ', 'A pedestrian in pink lower wear ', 'A pedestrian in purple lower wear', 
            # 'A pedestrian in red lower wear', 'A pedestrian in white lower wear', 'A pedestrian in yellow lower wear', 
            
            # 'A pedestrian in black shoes', 'A pedestrian in blue shoes', 'A pedestrian in brown shoes', 'A pedestrian in green shoes ', 
            # 'A pedestrian in greyshoes', 'A pedestrian in a orange shoes ', 'A pedestrian in pink shoes ', 'A pedestrian in purple shoes', 
            # 'A pedestrian in red shoes', 'A pedestrian in white shoes', 'A pedestrian in yellow shoes'
            ]

attr_words_neg = [
            'A pedestrian wearing no hat', 'A pedestrian wearing no muffler', 'A pedestrian with the headwear', 'A pedestrian wearing no sunglasses', 'A pedestrian with short hair',
            
            'A pedestrian in formal upper wear', 'A pedestrian in casual upper wear', 'A pedestrian without a jacket', 'A pedestrian in upper wear without a logo', 'A pedestrian without plaid upper wear',
            'A pedestrian without short-sleeved top', 'A pedestrian without upper wear with thin stripes', 'A pedestrian without a t-shirt', 'A pedestrian without other upper wear', 'A pedestrian A pedestrian without other upper wear upper wear with a V-neck',
            'A pedestrian in formal lower wear', 'A pedestrian in casual lower wear', 'A pedestrian without jeans', 'A pedestrian without shorts', 'A pedestrian without a short skirt', 'A pedestrian without trousers',
            'A pedestrian without leather shoes', 'A pedestrian without sandals', 'A pedestrian without other types of shoes', 'A pedestrian without sneakers',

            'A pedestrian without a backpack', 'A pedestrian without other types of attachments', 'A pedestrian without a messenger bag', 'A pedestrian with attachments', 'A pedestrian without plastic bags',
            'A pedestrian over the age of 30 ', 'A pedestrian under the age of 30 or over the age of 45', 'A pedestrian under the age of 45 or over the age of 60', 'A pedestrian under the age of 60',
            'A female pedestrian'#,
            
            # 'A pedestrian without black upper wear', 'A pedestrian without blue upper wear', 'A pedestrian without brown upper wear', 'A pedestrian without green upper wear ', 
            # 'A pedestrian without greyupper wear', 'A pedestrian without a orange upper wear ', 'A pedestrian without pwithoutk upper wear ', 'A pedestrian without purple upper wear', 
            # 'A pedestrian without red upper wear', 'A pedestrian without white upper wear', 'A pedestrian without yellow upper wear', 
            
            # 'A pedestrian without black lower wear', 'A pedestrian without blue lower wear', 'A pedestrian without brown lower wear', 'A pedestrian without green lower wear ', 
            # 'A pedestrian without greylower wear', 'A pedestrian without a orange lower wear ', 'A pedestrian without pwithoutk lower wear ', 'A pedestrian without purple lower wear', 
            # 'A pedestrian without red lower wear', 'A pedestrian without white lower wear', 'A pedestrian without yellow lower wear', 
            
            # 'A pedestrian without black shoes', 'A pedestrian without blue shoes', 'A pedestrian without brown shoes', 'A pedestrian without green shoes ', 
            # 'A pedestrian without greyshoes', 'A pedestrian without a orange shoes ', 'A pedestrian without pwithoutk shoes ', 'A pedestrian without purple shoes', 
            # 'A pedestrian without red shoes', 'A pedestrian without white shoes', 'A pedestrian without yellow shoes'
            ]


class LTCC(BaseImageDataset):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    def __init__(self, dataset_root='data', dataset_filename='LTCC_ReID', verbose=True, **kwargs):
        self.dataset_dir = osp.join(dataset_root, dataset_filename,'Pad_datasets')
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.mtrain_dir = osp.join(self.dataset_dir, 'mask/train')
        self.mquery_dir = osp.join(self.dataset_dir, 'mask/query')
        self.mgallery_dir = osp.join(self.dataset_dir, 'mask/test')
        self.meta_dir='PAR_PETA_1220_0.5.txt'
        self._check_before_run()
        self.meta_dims=35 #68
        
        imgdir2attribute = {}
        
        with open(osp.join(dataset_root, dataset_filename,self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present) 
        
        pid2label, attr2label,clothes2label =self.get_pid2label_and_clothes2label(self.train_dir,imgdir2attribute)
        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes,cam = self._process_dir_train(self.train_dir,imgdir2attribute,attr2label,clothes2label )#
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = self._process_dir_test(self.query_dir, self.gallery_dir,imgdir2attribute)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        if verbose:
            print("=> LTCC loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
            # print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
            print("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
            print("  ----------------------------------------")
            print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  ----------------------------------------")

        
        # attrlb=[]
        # for i in range(len(train)):
        #     attr = train[i][2]
        #     att = attr#[:30]
        #     attrlb.append(''.join(map(str,att)))
        # des_label=[]
        # des_refine=[]
        # lind=0
        # for d in attrlb:
        #     if d not in des_refine:
        #         des_refine.append(d)
        #         dlabel = lind
        #         lind += 1
        #     else:
        #         dlabel = des_refine.index(d)
        #     des_label.append(dlabel)
        # des_label = np.array(des_label)
        # # des_non = np.array(des_refine)
        # ds_train=[]
        # for i in range(len(train)):
        #     ml=list(train[i])
        #     ml[2]=des_label[i]
        #     train[i]=ml
            
        self.train = train
        self.query = query
        self.gallery = gallery
        self.cam = len(cam)

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path,imgdir2attribute, attr2label,clothes2label ):#
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        # clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            # clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            # clothes_container.add(clothes_id)
        pid_container = sorted(list(pid_container))
        # clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes2label)#clothes_container
        
        # num_pids = len(pid2label)
        # num_clothes = len(clothes2label)
        cam = set()
        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_dir in img_paths:
            pid, _, camid = map(int, pattern1.search(img_dir).groups())
            cam.add(camid)
            mask_path = osp.join(self.mtrain_dir, osp.basename(img_dir)[:-4]+ '.png')
            
            attr=imgdir2attribute[img_dir]
            attlb = []
            for i in range(len(attr)):
                att =  str(pid) + str(attr[i])#str(i)+
                attr_id = attr2label[i][att]
                attlb.append(attr_id)
            # attrlb=''.join(map(str, attr))
            # clothes =  str(pid) +attrlb#
            # pid = pid2label[pid]
            clolb=''.join(map(str, attr))
            clothes =  str(pid) +clolb#
            clothes_id = clothes2label[clothes]
            
            # clothes = pattern2.search(img_dir).group(1)
            camid -= 1  # index starts from 0
            pid = pid2label[pid]
            # clothes_id = clothes2label[clothes]
            ########################
            description =cap_gen_attr(imgdir2attribute[img_dir])
            dataset.append((img_dir, pid, clothes_id, camid,np.array(attlb),description,'','',mask_path))#des_inv,des_cloth,clothes_id,
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes,cam

    def _process_dir_test(self, query_path, gallery_path,imgdir2attribute):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        for img_dir in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_dir).groups())
            mask_path = osp.join(self.mquery_dir, osp.basename(img_dir)[:-4]+ '.png')
            clothes_id = pattern2.search(img_dir).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            ########################
            description,des_inv,des_cloth=cap_gen(imgdir2attribute[img_dir])
            query_dataset.append((img_dir, pid, clothes_id, camid, torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth,mask_path))

        for img_dir in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_dir).groups())
            mask_path = osp.join(self.mgallery_dir, osp.basename(img_dir)[:-4]+ '.png')
            clothes_id = pattern2.search(img_dir).group(1)
            camid -= 1 # index starts from 0
            clothes_id = clothes2label[clothes_id]
            ########################
            description,des_inv,des_cloth=cap_gen(imgdir2attribute[img_dir])
            gallery_dataset.append((img_dir, pid, clothes_id, camid, torch.tensor(imgdir2attribute[img_dir]).numpy(),description,des_inv,des_cloth,mask_path))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes
    
    def get_pid2label_and_clothes2label(self, dir_path,imgdir2attribute):
        
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set() 
        attr_container =[]
        for i in range(35):
            attr_container.append(set())
        # attr_container = set()    
        

        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            # clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            attr=imgdir2attribute[img_path]
            
            
            for i in range(len(attr)):
                att = str(pid) + str(attr[i])# str(i)+
                attr_container[i].add(att)
                
            attrlb=''.join(map(str, attr))
            clothes =  str(pid) +attrlb
            clothes_container.add(clothes)   
             
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        attr2label =[]
        for i in range(len(attr_container)):
            attr_container[i] = sorted(attr_container[i])
            attr2label.append( {attr_id:label for label, attr_id in enumerate(attr_container[i])})
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        # clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}    
            
    
        # num_pids = len(pid_container)
        # num_clothes = len(clothes_container)
        # pid2clothes = np.zeros((num_pids, num_clothes))
        # for img_path in img_paths:
        #     pid, _, _ = map(int, pattern1.search(img_path).groups())
            
        #     attr=imgdir2attribute[img_path]
        #     attrlb=''.join(map(str, attr))
        #     clothes =  str(pid) +attrlb#
        #     pid = pid2label[pid]
        #     clothes_id = clothes2label[clothes]
        #     pid2clothes[pid, clothes_id] = 1

        return pid2label, attr2label,clothes2label     
    
def reassignAttrlabel(pids,attribute):
        
        attr_container =[]
        for i in range(35):
            attr_container.append(set())  
        
        for i in range(len(pids)):
            pid = pids[i]
            # clothes_id = pattern2.search(osp.basename(img_path)).group(1)
            attr=attribute[i]
            if -1 in attr: continue
            for j in range(len(attr)):
                att = str(pid) + str(attr[j])
                attr_container[j].add(att)
                
        attr2label =[]
        for i in range(len(attr_container)):
            attr_container[i] = sorted(attr_container[i])
            attr2label.append( {attr_id:label for label, attr_id in enumerate(attr_container[i])})

        return attr2label 
def cap_gen_attr(attrlb):
    description=[]
    des_inv=''
    for i in range(len(attrlb)):
        if attrlb[i]==1:
            description.append(attr_words[i]) 
        else:
            description.append(attr_words_neg[i])
    description = np.array(description)
    return description  


def cap_gen(attrlb):
    description=''
    des_inv=''
    
    if attrlb[34]==1:
        description +="The "+ 'man' 
    else:
        description +="The "+ 'woman' 
        
    # des_cloth="A pedestrian with "  
        
    age=[" under the age of 30", " between the ages of 30 and 45",
            " between the ages of 45 and 60", " over the age of 60"]
    ageind=0
    for i in range(30,34):
        if  attrlb[i]==1:
            if ageind>0:
                description += ' or'
                # des_cloth += ' or'
            description += age[i-30]
            # des_cloth += age[i-30]
            ageind+=1
    if ageind==0:
        assert(" unknown years old")
        
    des_inv=description
    
    des_cloth = 'A pedestrain with '   
    description +=", with " 
    # des_cloth +=", with "   
    if attrlb[4]==1:
        description += "long "
        des_cloth += "long "
    else:
        description += "short "
        des_cloth += "short "
    
    description +="hair"
    # des_inv += "hair."
    des_cloth +="hair, "
    
    
    
    carry=["backpack", "other types of attachments",  "messenger bag",  "no attachments","plastic bag"]
    carryind=0
    for i in range(25,30):
        if  attrlb[i]==1:
            description += ", "
            description += carry[i-25]
            if carryind > 1:
                des_cloth += ", "
            des_cloth += carry[i-25]
            carryind+=1
    if carryind==0:
        description +=", unknown carrying"
        # description +=carry[3]
        des_cloth += "unknown carrying"
        
    
    headwear=[ "hat", "muffler", "no headwear","sunglasses"]
    headind=0
    for i in range(4):
        if  attrlb[i]==1:
            
            description += ", "
            des_cloth += ", "
            description += headwear[i]
            des_cloth += headwear[i]
            headind+=1
    if headind==0:
        description +="unknown headwear"
        des_cloth += "unknown headwear"
    description +=", "
    des_cloth += ", "
        
    if attrlb[5]==1:
        description += "casual upper wear"
        des_cloth += "casual upper wear"
    elif attrlb[6]==1:
        description += "formal upper wear"
        des_cloth += "formal upper wear"
    else:
        description += "uunknown style's upper wear"
        des_cloth += "unknown style's upper wear"
    
    upercloth=["jacket", "logo upper wear", "plaid upper wear", "short sleeves",  "thin stripes upper wear","t-shirt", "other upper wear","vneck upper wear"]
    uperwear=0
    for i in range(7,15):
        if  attrlb[i]==1:
            description +=', '
            description += upercloth[i-7]
            des_cloth +=', '
            des_cloth += upercloth[i-7] 
            uperwear+=1
    if uperwear==0:
        description += ", unknown upper wear" 
        des_cloth += ", unknown upper wear" 
    
    description += ", " 
    des_cloth += ", " 
    #lowbody    
    if attrlb[15]==1:
        description += "casual lower wear"
        des_cloth += "casual lower wear"
    elif attrlb[16]==1:
        description += "formal lower wear"
        des_cloth += "formal lower wear"
    else:
        description += "unknown style's lower wear" 
        des_cloth += "unknown style's lower wear"
    
            
    lowercloth=["jeans","shorts","short skirt","trousers"]
    lowerwear=0
    for i in range(17,21):
        if  attrlb[i]==1:
            description += ", " 
            des_cloth += ", " 
            description += lowercloth[i-17]
            des_cloth += lowercloth[i-17]
            lowerwear+=1
    if lowerwear==0:
        description +=", unknown lower wear" 
        des_cloth += ", unknown lower wear" 
    # description += ", and "
    # des_cloth += ", and "
    #footwear
    footwear=["leather shoes","sandals",'other types of shoes', "sneaker"]#["leatherShoes","sandals", "sneaker",'shoes']
    footind=0
    for i in range(21,25):
        if  attrlb[i]==1:
            description += ", " 
            des_cloth += ", " 
            description += footwear[i-21]
            des_cloth += footwear[i-21]
            footind+=1
    if footind==0:
        description +="unknown shoes"
        des_cloth +="unknown shoes" 
        
    description += "."
    des_cloth +="." 
    description = pre_caption(description,77)
    des_inv = pre_caption(des_inv,77)
    des_cloth = pre_caption(des_cloth,77)
    return description,des_inv,des_cloth


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption