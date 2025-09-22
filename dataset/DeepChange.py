import numpy as np
import os.path as osp
import re
import torch
import glob
attr_words = [
            'A pedestrian wearing a hat', 'A pedestrian wearing a muffler', 'A pedestrian with no headwear', 'A pedestrian wearing sunglasses', 'A pedestrian with long hair',
            
            'A pedestrian in casual upper wear', 'A pedestrian in formal upper wear', 'A pedestrian in a jacket', 'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
            'A pedestrian in a short-sleeved top', 'A pedestrian in upper wear with thin stripes', 'A pedestrian in a t-shirt', 'A pedestrian in other upper wear', 'A pedestrian in upper wear with a V-neck',
            'A pedestrian in casual lower wear', 'A pedestrian in formal lower wear', 'A pedestrian in jeans', 'A pedestrian in shorts', 'A pedestrian in a short skirt', 'A pedestrian in trousers',
            'A pedestrian in leather shoes', 'A pedestrian in sandals', 'A pedestrian in other types of shoes', 'A pedestrian in sneakers',

            'A pedestrian with a backpack', 'A pedestrian with other types of attachments', 'A pedestrian with a messenger bag', 'A pedestrian with no attachments', 'A pedestrian with plastic bags',
            'A pedestrian under the age of 30', 'A pedestrian between the ages of 30 and 45', 'A pedestrian between the ages of 45 and 60', 'A pedestrian over the age of 60',
            'A male pedestrian']

attr_words_neg = [
            'A pedestrian wearing no hat', 'A pedestrian wearing no muffler', 'A pedestrian with the headwear', 'A pedestrian wearing no sunglasses', 'A pedestrian with short hair',
            
            'A pedestrian in formal upper wear', 'A pedestrian in casual upper wear', 'A pedestrian without a jacket', 'A pedestrian in upper wear without a logo', 'A pedestrian without plaid upper wear',
            'A pedestrian without short-sleeved top', 'A pedestrian without upper wear with thin stripes', 'A pedestrian without a t-shirt', 'A pedestrian without other upper wear', 'A pedestrian A pedestrian without other upper wear upper wear with a V-neck',
            'A pedestrian in formal lower wear', 'A pedestrian in casual lower wear', 'A pedestrian without jeans', 'A pedestrian without shorts', 'A pedestrian without a short skirt', 'A pedestrian without trousers',
            'A pedestrian without leather shoes', 'A pedestrian without sandals', 'A pedestrian without other types of shoes', 'A pedestrian without sneakers',

            'A pedestrian without a backpack', 'A pedestrian without other types of attachments', 'A pedestrian without a messenger bag', 'A pedestrian with attachments', 'A pedestrian without plastic bags',
            'A pedestrian over the age of 30 ', 'A pedestrian under the age of 30 or over the age of 45', 'A pedestrian under the age of 45 or over the age of 60', 'A pedestrian under the age of 60',
            'A female pedestrian']

class DeepChange(object):
    """ DeepChange

    Reference:
        Xu et al. DeepChange: A Long-Term Person Re-Identification Benchmark. arXiv:2105.14685, 2021.

    URL: https://github.com/PengBoXiangShang/deepchange
    """
    def __init__(self, dataset_root='data', dataset_filename='DeepChangeDataset', verbose=True, **kwargs):
        self.dataset_dir = osp.join(dataset_root, dataset_filename,'Pad_datasets')
        # self.dataset_dir = osp.join(dataset_root, dataset_filename)
        self.train_dir = osp.join(self.dataset_dir, 'train-set')
        self.train_list = osp.join(self.dataset_dir, 'train-set-bbox.txt')
        self.val_query_dir = osp.join(self.dataset_dir, 'val-set-query')
        self.val_query_list = osp.join(self.dataset_dir, 'val-set-query-bbox.txt')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val-set-gallery')
        self.val_gallery_list = osp.join(self.dataset_dir, 'val-set-gallery-bbox.txt')
        self.test_query_dir = osp.join(self.dataset_dir, 'test-set-query')
        self.test_query_list = osp.join(self.dataset_dir, 'test-set-query-bbox.txt')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test-set-gallery')
        self.test_gallery_list = osp.join(self.dataset_dir, 'test-set-gallery-bbox.txt')
        
        self.mtrain_dir = osp.join(self.dataset_dir, 'mask/train-set')
        self.mtest_query_dir = osp.join(self.dataset_dir, 'mask/test-set-query')
        self.mtest_gallery_dir = osp.join(self.dataset_dir, 'mask/test-set-gallery')
        self.mval_query_dir = osp.join(self.dataset_dir, 'mask/val-set-query')
        self.mval_gallery_dir = osp.join(self.dataset_dir, 'mask/val-set-gallery')
        self.meta_dir='PAR_PETA_0220_0.5.txt'
        self._check_before_run()
        self.meta_dims=35
        imgdir2attribute = {}
        with open(osp.join(dataset_root, dataset_filename,self.meta_dir), 'r') as f:
            for line in f:
                imgdir, attribute_id, is_present = line.split()
                if imgdir not in imgdir2attribute:
                    imgdir2attribute[imgdir] = [0 for i in range(self.meta_dims)]
                imgdir2attribute[imgdir][int(attribute_id)] = int(is_present) 


        train_names = self._get_names(self.train_list)
        val_query_names = self._get_names(self.val_query_list)
        val_gallery_names = self._get_names(self.val_gallery_list)
        test_query_names = self._get_names(self.test_query_list)
        test_gallery_names = self._get_names(self.test_gallery_list)

        pid2label, clothes2label,attr2label, pid2clothes = self.get_pid2label_and_clothes2label(train_names,self.train_dir,imgdir2attribute)#
        train, num_train_pids, num_train_clothes = self._process_dir(self.train_dir, train_names, clothes2label, attr2label,imgdir2attribute, pid2label=pid2label)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(val_query_names,self.val_query_dir,imgdir2attribute,val_gallery_names)#
        val_query, num_val_query_pids, num_val_query_clothes  = self._process_dir(self.val_query_dir, val_query_names, clothes2label, imgdir2attribute=imgdir2attribute)
        val_gallery, num_val_gallery_pids, num_val_gallery_clothes = self._process_dir(self.val_gallery_dir, val_gallery_names, clothes2label, imgdir2attribute=imgdir2attribute)
        num_val_pids = len(pid2label)
        num_val_clothes = len(clothes2label)

        pid2label, clothes2label = self.get_pid2label_and_clothes2label(test_query_names,self.test_query_dir,imgdir2attribute, test_gallery_names)#
        test_query, num_test_query_pids, num_test_query_clothes = self._process_dir(self.test_query_dir, test_query_names, clothes2label, imgdir2attribute=imgdir2attribute)
        test_gallery, num_test_gallery_pids, num_test_gallery_clothes = self._process_dir(self.test_gallery_dir, test_gallery_names, clothes2label, imgdir2attribute=imgdir2attribute)
        num_test_pids = len(pid2label)
        num_test_clothes = len(clothes2label)

        num_total_pids = num_train_pids + num_val_pids + num_test_pids
        num_total_clothes = num_train_clothes + num_val_clothes + num_test_clothes
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)

        if verbose:
            print("=> DeepChange loaded")
            print("Dataset statistics:")
            print("  --------------------------------------------")
            print("  subset        | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train         | {:5d} | {:8d} | {:9d} ".format(num_train_pids, len(train), num_train_clothes))
            print("  query(val)    | {:5d} | {:8d} | {:9d} ".format(num_val_query_pids, len(val_query), num_val_query_clothes))
            print("  gallery(val)  | {:5d} | {:8d} | {:9d} ".format(num_val_gallery_pids, len(val_gallery), num_val_gallery_clothes))
            print("  query         | {:5d} | {:8d} | {:9d} ".format(num_test_query_pids, len(test_query), num_test_query_clothes))
            print("  gallery       | {:5d} | {:8d} | {:9d} ".format(num_test_gallery_pids, len(test_gallery), num_test_gallery_clothes))
            print("  --------------------------------------------")
            print("  total         | {:5d} | {:8d} | {:9d} ".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  --------------------------------------------")
        
        
            
        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def get_pid2label_and_clothes2label(self, img_names1,homedir,imgdir2attribute, img_names2=None):
        if img_names2 is not None:
            img_names = img_names1 + img_names2
        else:
            img_names = img_names1

        pid_container = set()
        attr_container =[]
        for i in range(35):
            attr_container.append(set())
        clothes_container = set()
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid_container.add(pid)
            if img_names2 is None:
                img_path = osp.join(homedir, img_name.split(',')[0])
                
                attr=imgdir2attribute[img_path]
                for i in range(len(attr)):
                    att = str(pid) + str(attr[i])
                    attr_container[i].add(att)
                
            clothes_container.add(clothes)
        pid_container = sorted(list(pid_container))
        
        if img_names2 is None:
            attr2label =[]
            for i in range(len(attr_container)):
                attr_container[i] = sorted(attr_container[i])
                attr2label.append( {attr_id:label for label, attr_id in enumerate(attr_container[i])})
        
        clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        if img_names2 is not None:
            return pid2label, clothes2label#

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, attr2label, pid2clothes
    
    def get_pid2label_and_clothes2label_1(self, img_names1,homedir,imgdir2attribute, img_names2=None):
        if img_names2 is not None:
            img_names = img_names1 + img_names2
        else:
            img_names = img_names1

        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            img_path = osp.join(homedir, img_name.split(',')[0])
            # clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid_container.add(pid)
            
            attr=imgdir2attribute[img_path]
            attrlb=''.join(map(str, attr))
            clothes = names[0] + attrlb
            
            clothes_container.add(clothes)
        pid_container = sorted(list(pid_container))
        clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        if img_names2 is not None:
            return pid2label, clothes2label

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_name in img_names:
            names = img_name.split('.')[0].split('_')
            
            img_path = osp.join(homedir, img_name.split(',')[0])
            attr=imgdir2attribute[img_path]
            attrlb=''.join(map(str, attr))
            clothes = names[0] + attrlb
            # clothes = names[0] + names[2]
            pid = int(names[0][1:])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_query_dir):
            raise RuntimeError("'{}' is not available".format(self.val_query_dir))
        if not osp.exists(self.val_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.val_gallery_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.test_gallery_dir))

    def _process_dir(self, home_dir, img_names, clothes2label, attr2label=None,imgdir2attribute=None, pid2label=None):
        dataset = []
        pid_container = set()
        clothes_container = set()
        for img_name in img_names:
            img_path = osp.join(home_dir, img_name.split(',')[0])
            names = img_name.split('.')[0].split('_')
            tracklet_id = int(img_name.split(',')[1])
            
            if pid2label is not None:
                mask_path = osp.join(self.mtrain_dir, osp.basename(img_path)[:-4]+ '.png')
            else:
                mask_path=''
            
            # attrlb=''.join(map(str, attr))
            # clothes = names[0] + attrlb
            clothes = names[0] + names[2]
            clothes_id = clothes2label[clothes]
            clothes_container.add(clothes_id)
            pid = int(names[0][1:])
            pid_container.add(pid)
            
            attr=imgdir2attribute[img_path]
            if pid2label is not None:
                attlb = []
                for i in range(len(attr)):
                    att = str(pid) + str(attr[i])
                    if attr2label is not None:
                        attr_id = attr2label[i][att]
                    else:
                        attr_id = att
                    attlb.append(attr_id)
                attlb = np.array(attlb)
            else:
                attlb=torch.tensor(imgdir2attribute[img_path]).numpy()    
            camid = int(names[1][1:])
            if pid2label is not None:
                pid = pid2label[pid]
            description=self.cap_gen_attr(imgdir2attribute[img_path])
            des_inv,des_cloth='',''
            # on DeepChange, we allow the true matches coming from the same camera 
            # but different tracklets as query following the original paper.
            # So we use tracklet_id to replace camid for each sample.
            dataset.append((img_path, pid, clothes_id,tracklet_id,np.array(attlb),description,des_inv,des_cloth,mask_path))# camid
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        return dataset, num_pids, num_clothes
    
    def cap_gen_attr(self,attrlb):
        description=[]
        des_inv=''
        for i in range(len(attrlb)):
            if attrlb[i]==1:
                description.append(attr_words[i]) 
            else:
                description.append(attr_words_neg[i])
        description = np.array(description)
        return description 

    def cap_gen(self,attrlb):
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
    
# for k in range(num_train_pids):
        #     tr_pid=[]
        #     indx=[]
        #     for i in range(len(train)):
        #         if train[i][1]==k:
        #             tr_pid.append(train[i])
        #             indx.append(i)
                
        #     attrlb=[]
        #     for ds in tr_pid:
        #         attrlb.append(''.join(map(str,ds[4])))
        #     des_label=[]
        #     des_refine=[]
        #     lind=0
        #     for d in attrlb:
        #         if d not in des_refine:
        #             des_refine.append(d)
        #             dlabel = str(k) +'_' +  str(lind)
        #             lind += 1
        #         else:
        #             dlabel = des_refine.index(d)
        #         des_label.append(dlabel)
        #     des_label = np.array(des_label)
        #     # des_non = np.array(des_refine)
        #     ds_train=[]
        #     for j in range(len(tr_pid)):
        #         ml=list(train[indx[j]])
        #         ml[4]=des_label[j]
        #         train[indx[j]]=ml
        
        # attrlb1=[]
        # for i in range(len(train)):
        #     attrlb1.append(''.join(map(str,train[i][4])))