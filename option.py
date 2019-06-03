# -*- coding: utf-8 -*-
import argparse
import math
parser = argparse.ArgumentParser(description='FRCNN')


parser.add_argument('--lr', type=float, default=1e-5,
                    help = 'learning rate')
parser.add_argument('--batch_size', type=int, default=1,
                    help = 'input batch size for learning')
parser.add_argument('--decay', type = str, default='200',
                    help='learning rate decay type')
parser.add_argument('--epochs', type =int, default=10,
                    help='number of epochs to train')
parser.add_argument('--num_every', type =int, default=50,
                    help='number of times in one epoch to train')
parser.add_argument('--img_width', type =int, default=300,
                    help='resize img width')
parser.add_argument('--img_height', type =int, default=120,
                    help='resize img height')

parser.add_argument('--test_only', action='store_true', default= False,
                    help='set this option to test the model')
parser.add_argument('--test_img_path', type = str, default='./test_imgs',
                    help='save path for the training model')

parser.add_argument('--model', default='ResNet50',
                    choices=('ResNet50'),
                    help='choose a classification model')
parser.add_argument('--config_filename', type = str, default='defatule_config',
                    help='path for the config_filename (train for save/ test fot load)')
parser.add_argument('--model_save_path', type = str, default='./frcnn_model.hdf5',
                    help='save path for the training model')
parser.add_argument('--pretrained_model', type = str, default='./frcnn_model.hdf5',
                    help='path for the pre_training model')
parser.add_argument('--VOC_path', type = str, default='../private_project/SiamGAN/VOC2012',
                    help= 'train_image_path include  Annotations, ImageSets and JPEGImages ')


parser.add_argument('--n_dims', type =int, default=1,
                    help='RGB = 3, Gray = 1')


args = parser.parse_args()

args.anchor_box_scales = [ 32, 64, 128]
args.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
args.num_rois = 32
args.rpn_max_overlap = 0.7  #训练时RPN_iou>0.7的框会被认为是正样本
args.rpn_min_overlap = 0.3  #训练时RPN_iou<0.3的框会被认为是负样本

args.classifier_min_overlap = 0.1 #分类的阈值
args.classifier_max_overlap = 0.5

# ???
args.std_scaling = 4
args.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

if args.model == "ResNet50":
    args.rpn_stride = 16   #下降的倍数  ResNet50为16

