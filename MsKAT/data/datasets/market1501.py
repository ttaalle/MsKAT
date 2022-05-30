# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp
import pdb
import scipy.io as scio
import mat4py

from .bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        self.att_dir =osp.join(self.dataset_dir,'market_attribute.mat')
        self.attribute = mat4py.loadmat(self.att_dir)['market_attribute']
        self.train_attr = self.attribute['train']  # attribute in train dataset

        # self.image_index = self.train_attr['image_index']
        self.age = self.train_attr['age']
        self.backpack = self.train_attr['backpack']
        self.bag = self.train_attr['bag']
        self.handbag = self.train_attr['handbag']
        self.downblack = self.train_attr['downblack']
        self.downblue = self.train_attr['downblue']
        self.downbrown = self.train_attr['downbrown']
        self.downgray = self.train_attr['downgray']
        self.downgreen = self.train_attr['downgreen']
        self.downpink = self.train_attr['downpink']
        self.downpurple = self.train_attr['downpurple']
        self.downwhite = self.train_attr['downwhite']
        self.downyellow = self.train_attr['downyellow']
        self.upblack = self.train_attr['upblack']
        self.upblue = self.train_attr['upblue']
        self.upgreen = self.train_attr['upgreen']
        self.upgray = self.train_attr['upgray']
        self.uppurple = self.train_attr['uppurple']
        self.upred = self.train_attr['upred']
        self.upwhite = self.train_attr['upwhite']
        self.upyellow = self.train_attr['upyellow']
        self.clothes = self.train_attr['clothes']
        self.down = self.train_attr['down']
        self.up = self.train_attr['up']
        self.hair = self.train_attr['hair']
        self.hat = self.train_attr['hat']
        self.gender = self.train_attr['gender']


        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams= self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams= self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams= self.get_imagedata_info(self.gallery)

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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: 
                pid = pid2label[pid]
                # ------- age -----------
                age = self.age[pid]
                if age == 1:
                    age = 0
                elif age == 2:
                    age = 1
                elif age == 3:
                    age = 2
                else:
                    age = 3
                # ------- backpack -----------
                backpack = self.backpack[pid]
                if backpack == 2:
                    backpack = 1
                else:
                    backpack = 0
                # ------- bag -----------
                bag = self.bag[pid]
                if bag == 2:
                    bag = 1
                else:
                    bag = 0
                # ------- handbag -----------
                handbag = self.handbag[pid]
                if handbag == 2:
                    handbag = 1
                else:
                    handbag = 0
                # ------- down color -----------
                downblack = self.downblack[pid]
                downblue = self.downblue[pid]
                downbrown = self.downbrown[pid]
                downgray = self.downgray[pid]
                downgreen = self.downgreen[pid]
                downpink = self.downpink[pid]
                downpurple = self.downpurple[pid]
                downwhite = self.downwhite[pid]
                downyellow = self.downyellow[pid]
                if downblack == 2:
                    down_color = 0
                elif downblue == 2:
                    down_color = 1
                elif downbrown == 2:
                    down_color = 2
                elif downgray == 2:
                    down_color = 3
                elif downgreen == 2:
                    down_color = 4
                elif downpink == 2:
                    down_color = 5
                elif downpurple == 2:
                    down_color = 6
                elif downwhite == 2:
                    down_color = 7
                elif downyellow == 2:
                    down_color = 8

                # ------- up color -----------
                upblack = self.upblack[pid]
                upblue = self.upblue[pid]
                upgreen = self.upgreen[pid]
                upgray = self.upgray[pid]
                uppurple = self.uppurple[pid]
                upred = self.upred[pid]
                upwhite = self.upwhite[pid]
                upyellow = self.upyellow[pid]
                if upblack == 2:
                    up_color = 0
                elif upblue == 2:
                    up_color = 1
                elif upgreen == 2:
                    up_color = 2
                elif upgray == 2:
                    up_color = 3
                elif uppurple == 2:
                    up_color = 4
                elif upred == 2:
                    up_color = 5
                elif upwhite == 2:
                    up_color = 6
                elif upyellow == 2:
                    up_color = 7
                # ------- clothes -----------
                clothes = self.clothes[pid]
                if clothes == 1:
                    clothes = 0
                else:
                    clothes = 1
                # ------- down -----------
                down = self.down[pid]
                if down == 1:
                    down = 0
                else:
                    down = 1
                # ------- up -----------
                up = self.up[pid]
                if up == 1:
                    up = 0
                else:
                    up = 1
                # ------- hair -----------
                hair = self.hair[pid]
                if hair == 1:
                    hair = 0
                else:
                    hair = 1
                # ------- hat -----------
                hat = self.hat[pid]
                if hat == 2:
                    hat = 1
                else:
                    hat = 0
                # ------- gender -----------
                gender = self.gender[pid]
                if gender == 1:
                    gender = 0
                else:
                    gender = 1
            else:
                age=0
                backpack=0
                bag=0
                handbag=0
                down_color=0
                up_color=0
                clothes=0
                down=0
                up=0
                hair=0
                hat=0
                gender=0

            dataset.append((img_path, pid, camid, age, backpack, bag, handbag, down_color, \
                            up_color, clothes, down, up, hair, hat, gender))
        return dataset
