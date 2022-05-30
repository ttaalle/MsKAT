# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, camid, age, backpack, bag, handbag, down_color, \
                            up_color, clothes, down, up, hair, hat, gender, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)

    camid = torch.tensor(camid, dtype=torch.int64)
    age = torch.tensor(age, dtype=torch.int64)
    backpack = torch.tensor(backpack, dtype=torch.int64)
    bag = torch.tensor(bag, dtype=torch.int64)
    handbag = torch.tensor(handbag, dtype=torch.int64)
    down_color = torch.tensor(down_color, dtype=torch.int64)
    up_color = torch.tensor(up_color, dtype=torch.int64)
    clothes = torch.tensor(clothes, dtype=torch.int64)
    down = torch.tensor(down, dtype=torch.int64)
    up = torch.tensor(up, dtype=torch.int64)
    hair = torch.tensor(hair, dtype=torch.int64)
    hat = torch.tensor(hat, dtype=torch.int64)
    gender = torch.tensor(gender, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camid, age, backpack, bag, handbag, down_color, \
                            up_color, clothes, down, up, hair, hat, gender


def val_collate_fn(batch):
    imgs, pids, camids, _ ,_,_,_,_,_,_,_,_,_,_,_,_= zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
