# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F
import pdb

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:(dataset.quer(dataset.query y 
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionAT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2):
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q= x_1 #B,C,N
        k= x_2 #B,C,N1
        v= x_2 #B,C,N1

        attn = (q.transpose(-2, -1) @ k)* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v.transpose(1, 2))
        #x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = self.proj(x)
        #x = self.proj_drop(x)
        return x

class AttentionST(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2):
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q= normalize(x_1.transpose(-1,-2))  #B,C,N
        k= normalize(x_2.transpose(-1,-2))  #B,C,N1
        v= x_2#B,C,N1

        attn = (k @ q.transpose(-2, -1))
        attn = torch.repeat_interleave(attn, k.shape[2], -1)
        
        x = (attn * v.transpose(-1, -2))
        #x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = self.proj(x)
        #x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap1 = nn.AdaptiveAvgPool1d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.in_planes1 = 2048
        self.in_planes2 = 256
        self.in_planes3 = 1024+2048
        self.in_planes4 = 1024
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.ln = nn.LayerNorm(256)
        
        self.conv_at = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_st = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_id1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_id2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_id3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0,
                               bias=True)
        self.conv_id4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.attn1 = AttentionST(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn2 = AttentionAT(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn11 = AttentionST(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn22 = AttentionAT(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn111 = AttentionST(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn222 = AttentionAT(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn1111 = AttentionST(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.attn2222 = AttentionAT(dim=256, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)

        self.mlp1 = Mlp(in_features=256, hidden_features=256*2, act_layer=nn.GELU, drop=0.)
        self.mlp2 = Mlp(in_features=256, hidden_features=256*2, act_layer=nn.GELU, drop=0.)
        self.mlp3 = Mlp(in_features=256, hidden_features=256*2, act_layer=nn.GELU, drop=0.)
        self.mlp4 = Mlp(in_features=256, hidden_features=256*2, act_layer=nn.GELU, drop=0.)

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck1 = nn.BatchNorm1d(self.in_planes3)
            self.bottleneck1.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier1 = nn.Linear(self.in_planes3, self.num_classes, bias=False)

            self.classifier_cam = nn.Linear(self.in_planes2, 6, bias=False) #cam (6 for market)

            self.classifier_age = nn.Linear(self.in_planes2, 4, bias=False)  #view(4 for market)
            self.classifier_backpack = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_bag = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_handbag = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_downcolor = nn.Linear(self.in_planes2, 9, bias=False)  #view(2 for market)
            self.classifier_upcolor = nn.Linear(self.in_planes2, 8, bias=False)  #view(2 for market)
            self.classifier_clothes = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_up = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_down = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_hair = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_hat = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)
            self.classifier_gender = nn.Linear(self.in_planes2, 2, bias=False)  #view(2 for market)


            #self.classifier_time = nn.Linear(self.in_planes2, 24, bias=False)  #time(1 for veri, 24 for wild)
            #self.classifier_model = nn.Linear(self.in_planes2, 153, bias=False) #moid (1 for veri, 153 for wild)
            self.classifier_color = nn.Linear(self.in_planes2, 10, bias=False)  #color(10 for veri,12 for wild)
            self.classifier_type = nn.Linear(self.in_planes2, 9, bias=False) #type(9 for veri, 14 for wild)

            self.bottleneck.apply(weights_init_kaiming)
            self.bottleneck1.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.classifier1.apply(weights_init_classifier)

            self.conv_at.apply(weights_init_kaiming)
            self.conv_st.apply(weights_init_kaiming)
            self.classifier_cam .apply(weights_init_classifier)

            self.classifier_age.apply(weights_init_classifier)
            self.classifier_backpack.apply(weights_init_classifier)  
            self.classifier_bag.apply(weights_init_classifier)
            self.classifier_handbag.apply(weights_init_classifier)
            self.classifier_downcolor.apply(weights_init_classifier)
            self.classifier_upcolor.apply(weights_init_classifier)

            self.classifier_clothes.apply(weights_init_classifier)
            self.classifier_up.apply(weights_init_classifier)
            self.classifier_down.apply(weights_init_classifier)
            self.classifier_hair.apply(weights_init_classifier)
            self.classifier_hat.apply(weights_init_classifier)
            self.classifier_gender.apply(weights_init_classifier)
            self.conv_id1.apply(weights_init_kaiming)
            self.conv_id2.apply(weights_init_kaiming)
            self.conv_id3.apply(weights_init_kaiming)
            self.conv_id4.apply(weights_init_kaiming)

    def forward(self, x):

        a, a3, a2, a1=self.base(x)#提取四层特征
        global_feat = self.gap(a)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        a_st =  self.conv_st(a) #状态卷积
        a_st = self.bn1(a_st)   #状态正则化
        a_st = self.relu(a_st) #状态relu

        a_at =  self.conv_st(a) #属性卷积
        a_at = self.bn1(a_at)   #属性正则化
        a_at = self.relu(a_at) #属性relu

        st_feat = self.gap(a_st)  # (b, 2048, 1, 1)
        st_feat = st_feat.view(st_feat.shape[0], -1)  # flatten to (bs, 2048)

        at_feat = self.gap(a_at)  # (b, 2048, 1, 1)
        at_feat = at_feat.view(at_feat.shape[0], -1)  # flatten to (bs, 2048)

        V1 = self.relu(self.bn1(self.conv_id1(a1)))  #[32, 256, 64, 64],第一层特征做trans映射
        V2 = self.relu(self.bn1(self.conv_id2(a2)))  #[32, 256, 32, 32]，第二层特征做trans映射
        V3 = self.relu(self.bn1(self.conv_id3(a3)))  #[32, 256, 16, 16]，第三层特征做trans映射
        V4 = self.relu(self.bn1(self.conv_id4(a)))   #[32, 256, 16, 16]，第四层特征做trans映射

        F_SeT1 = self.attn1(st_feat.unsqueeze(-1), V1.flatten(2)).transpose(-1,-2) #b,256,4096，第一层状态消除Trans
        R_1 = V1.flatten(2)-F_SeT1 #第一层残差
        R_1 = self.mlp1(self.ln(R_1.transpose(-1,-2))).transpose(-1,-2) #第一层LN+FFN
        F_AaT1= self.attn2(at_feat.unsqueeze(-1), R_1).transpose(-1,-2) #b,256,1，第一层属性集合trans
        f_s1 = self.gap1(F_SeT1).squeeze(-1) #对第一层状态消除的特征做gap
        f_a1 = F_AaT1.squeeze(-1) #获取第一层属性聚合特征

        F_SeT2 = self.attn11(st_feat.unsqueeze(-1), V2.flatten(2)).transpose(-1,-2) #b,256,4096，第二层
        R_2 = V2.flatten(2)-F_SeT2 #第二层
        R_2 = self.mlp2(self.ln(R_2.transpose(-1,-2))).transpose(-1,-2) #第二层
        F_AaT2= self.attn22(at_feat.unsqueeze(-1), R_2).transpose(-1,-2) #b,256,1，第二层
        f_s2 = self.gap1(F_SeT2).squeeze(-1) #第二层
        f_a2 = F_AaT2.squeeze(-1) #第二层

        F_SeT3 = self.attn111(st_feat.unsqueeze(-1), V3.flatten(2)).transpose(-1,-2) #b,256,4096，第三层
        R_3 = V3.flatten(2)-F_SeT3 #第三层
        R_3 = self.mlp3(self.ln(R_3.transpose(-1,-2))).transpose(-1,-2) #第三层
        F_AaT3= self.attn222(at_feat.unsqueeze(-1), R_3).transpose(-1,-2) #b,256,1，第三层
        f_s3 = self.gap1(F_SeT3).squeeze(-1) #第三层
        f_a3 = F_AaT3.squeeze(-1) #第三层

        F_SeT4 = self.attn1111(st_feat.unsqueeze(-1), V4.flatten(2)).transpose(-1,-2) #b,256,4096，第四层
        R_4 = V4.flatten(2)-F_SeT4 #第四层
        R_4 = self.mlp4(self.ln(R_4.transpose(-1,-2))).transpose(-1,-2) #第四层
        F_AaT4= self.attn2222(at_feat.unsqueeze(-1), R_4).transpose(-1,-2) #b,256,1，第四层
        f_s4 = self.gap1(F_SeT4).squeeze(-1) #第四层
        f_a4 = F_AaT4.squeeze(-1) #第四层

        f = torch.cat((f_a1,f_a2),dim=-1) 
        f = torch.cat((f,f_a3),dim=-1)
        f = torch.cat((f,f_a4),dim=-1)
        global_feat1 = torch.cat((f,global_feat),dim=-1) #将全局特征与属性聚合特征进行拼接



        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            feat1 = self.bottleneck1(global_feat1)  # normalize for angular softmax  

        if self.training:
            cls_score = self.classifier(feat)
            cls_score1 = self.classifier1(feat1)

            cls_cam = self.classifier_cam(st_feat)
            
            cls_age =self.classifier_age(at_feat)
            cls_backpack =self.classifier_backpack(at_feat)
            cls_bag = self.classifier_bag(at_feat)
            cls_handbag =self.classifier_handbag(at_feat)
            cls_downcolor= self.classifier_downcolor(at_feat)
            cls_upcolor =self.classifier_upcolor(at_feat)

            cls_clothes= self.classifier_clothes(at_feat)
            cls_up = self.classifier_up(at_feat)
            cls_down =self.classifier_down(at_feat)
            cls_hair =self.classifier_hair(at_feat)
            cls_hat = self.classifier_hat(at_feat)
            cls_gender =self.classifier_gender(at_feat)


            return cls_score, global_feat,cls_score1, global_feat1, cls_cam, cls_age, cls_backpack, cls_bag, cls_handbag, cls_downcolor, cls_upcolor, cls_clothes, cls_up, cls_down, cls_hair, cls_hat, cls_gender,f_s1, f_a1, f_s2, f_a2, f_s3, f_a3, f_s4, f_a4  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat1
            else:
                # print("Test with feature before BN")
                return global_feat1

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
