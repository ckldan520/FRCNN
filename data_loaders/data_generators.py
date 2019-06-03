# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random


#IOU计算两个框之间的IOU值
def iou(a, b):
    #a and b should be (x1, y1, x2, y2)
    if a[0]>=a[2] or a[1]>=a[3] or b[0]>=b[2] or b[1]>=b[3]:
        return 0.0

    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w<0 or h<0:
        return 0
    area_i = w*h
    area_u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area_i
    return float(area_i) / float(area_u + 1e-6)


#根据图像resize后大小求图像输出的大小
def get_img_output_length(args, width, height):
    def get_output_length(args, input_length):
        if args.model=='ResNet50':
            input_length += 6  #padding
            filter_sizes = [7, 3, 1, 1]  #4 strided convolutions
            stride = 2
            for filter_size in filter_sizes:
                input_length = (input_length - filter_size ) // stride + 1
            return input_length
        else:
            print("network is wrong !!! Program Interrupt")
            return 0
    return get_output_length(args, width), get_output_length(args, height)



#做标签数据
def calc_rpn(args, img_data, width, height, resized_width, resized_height):
    downscale = args.rpn_stride
    anchor_sizes = args.anchor_box_scales
    anchor_ratios = args.anchor_box_ratios
    num_anchors= len(anchor_ratios) * len(anchor_sizes)
    n_anchratios = len(anchor_ratios)


    #calculate the feature_extracter network output map_size based on the network architecture    ``
    (output_width, output_height) = get_img_output_length(args, resized_width, resized_height)

    #initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))  #记录每个锚点 是否大于了RPN阈值
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors)) #记录每个锚点 激活的状态
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors*4))   #记录每个锚点的 dx dy dw dh

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)  #记录每个框有多少个正样本的锚点
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int) #最符合GT的锚点 记录下 锚点y x 比例 尺度
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)  #IOU值最大的记录数据（对每个框而言）
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)      #符合GT的锚点的两个坐标
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)  #符合GT的锚点 dx dy dw dh

    #get the GT box coordinates and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))  #GT样本的位置点
    for  bbox_num, bbox in enumerate(img_data['bboxes']):
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]  #锚的宽度
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]  #锚的高度

            for ix in range(output_width):
                #x-coordinates of the current anchor box x坐标还原回原图中的大小
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2 #锚的左边界
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2 #锚的右边界

                #ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc >resized_width:
                    continue

                for jy in range(output_height):
                    # x-coordinates of the current anchor box yzuob还原回原图中的大小
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2  #锚的下边界
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2  #锚的上边界
                    #ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc >resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    #this is the best IOU for the (x,y) coord and the current anchor
                    #note that this is differnet from the best IOU for a GT box
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        #get IOU of the current GT_box and the current anchor box计算IOU
                        temp_a = [gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]]
                        temp_b = [x1_anc, y1_anc, x2_anc, y2_anc]
                        curr_iou = iou(temp_a, temp_b)
                        #calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou>args.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0 #GT 中心x坐标
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0 #GT 中心y坐标
                            cxa = (x1_anc + x2_anc) / 2.0 #Anchor  中心x坐标
                            cya = (y1_anc + y2_anc) / 2.0 #Anchor  中心y坐标

                            tx = (cx - cxa) / (x2_anc - x1_anc) #dx/ w
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            #all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            #anchor to positive if the IOU is > 0.7 设置为正样本
                            if curr_iou > args.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # update the regression layer target if this IOU is the best for the current(x, y)
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            #if the IOU > 0.3 and <0.7, it is ambiguous and no included in the objective
                            if args.rpn_min_overlap < curr_iou < args.rpn_max_overlap:
                                #gray_zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off opuputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios*anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios*anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios*anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

    #ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            #no box with an IOU greater than zero
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios* best_anchor_for_bbox[idx, 3]
            ] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]
            ] = 1
            start = 4*(best_anchor_for_bbox[idx, 2]+ n_anchratios*best_anchor_for_bbox[idx,3])
            y_rpn_regr[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:]==1, y_is_box_valid[0,:,:,:]==1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0,:,:,:]==0, y_is_box_valid[0,:,:,:]==1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some
    # of the negative regions. We also limit it to 256 regions
    num_regions = 256

    if len(pos_locs[0]) > num_regions /2 : #正样本控制在 128个 以下
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions/2
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


#数据整合成一个batch
def generate_batch_data(args, all_imgs, num_every_current):
    try:
        batch_size = args.batch_size
        start_index = (num_every_current * batch_size) % len(all_imgs)
        train_img_set = []
        for i in range(batch_size):
            index = (i + start_index) % len(all_imgs)
            train_img_set.append(all_imgs[index])
            img = cv2.imread(all_imgs[index]['filepath'])
            org_width = img.shape[1]
            org_height = img.shape[0]

            if args.n_dims == 1:
                img = img[:, :, 0]

            # resize the image to fixed size
            img = cv2.resize(img, (args.img_width, args.img_height), interpolation=cv2.INTER_CUBIC)

            y_rpn_cls, y_rpn_regr = calc_rpn(args, all_imgs[index], width=org_width, height=org_height, resized_width=args.img_width, resized_height= args.img_height)

            #img_data process 数据规则化为0-1之间 为4维（batch,通道数，宽度，高度）
            temp_max = np.max(img)
            img = img.astype(np.float32)
            img = img / temp_max
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)

            if i == 0:
                img_data_batch=img
                RPN_cls_label = y_rpn_cls
                RPN_reg_label = y_rpn_regr
            else:
                img_data_batch = np.concatenate([img_data_batch, img], axis=0)
                RPN_cls_label = np.concatenate([RPN_cls_label, y_rpn_cls], axis=0)
                RPN_reg_label = np.concatenate([RPN_reg_label, y_rpn_regr], axis=0)

            RPN_reg_label[:, RPN_reg_label.shape[1] // 2:, :, :] *= args.std_scaling

        #tensorflow 维度顺序为 batch,宽度，高度，通道数
        img_data_batch = np.transpose(img_data_batch, (0, 2, 3, 1))
        RPN_cls_label = np.transpose(RPN_cls_label, (0, 2, 3, 1))
        RPN_reg_label = np.transpose(RPN_reg_label, (0, 2, 3, 1))

        return img_data_batch, [RPN_cls_label, RPN_reg_label], train_img_set
    except Exception as e:
        print('generate_batch_data wrong')


def generate_test_data(args, test_img):
    img = cv2.imread(test_img['filepath'])
    org_width = img.shape[1]
    org_height = img.shape[0]

    if args.n_dims == 1:
        img = img[:, :, 0]

    # resize the image to fixed size
    img = cv2.resize(img, (args.img_width, args.img_height), interpolation=cv2.INTER_CUBIC)

    # img_data process 数据规则化为0-1之间 为4维（batch,通道数，宽度，高度）
    temp_max = np.max(img)
    img = img.astype(np.float32)
    img = img / temp_max
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    img = np.transpose(img, (0, 2, 3, 1))

    return img, [org_width, org_height]





