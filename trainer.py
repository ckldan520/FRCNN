# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import keras
import cv2
import roi_helper2 as roi_helper
from data_loaders import data_generators as dt_gen
from keras.utils import generic_utils
import time


class Trainer():
    def __init__(self, args, loader, network):
        self.args = args
        self.network = network
        self.loader = loader
        self.epoch_current = 0
        self.best_loss = np.Inf

    def train(self):
        self.epoch_current += 1
        progbar = generic_utils.Progbar(self.args.num_every)
        print('Epoch {}/{}'.format(self.epoch_current, self.args.epochs))
        num_every_current = 0
        losses = np.zeros((self.args.num_every, 5))
        temp_train_imgs = self.loader.train_imgs
        np.random.shuffle(temp_train_imgs)

        try:
            start_time = time.time()

            while num_every_current < self.args.num_every:
                X, Y, train_img_set = dt_gen.generate_batch_data(self.args, temp_train_imgs, num_every_current)

                loss_rpn = self.network.rpn_model.train_on_batch(X, Y)
                Pred_rpn = self.network.rpn_model.predict_on_batch(X)

                R = roi_helper.rpn_to_roi(self.args, Pred_rpn[0], Pred_rpn[1], use_regr=True, overlap_thresh=0.7, max_boxes=300)
                X2, Y1, Y2, _ = roi_helper.calc_iou(self.args, R, train_img_set[0], self.loader.class_mapping)

                #X2, Y1, Y2 = roi_helper.classify_label_convert(self.args, Pred_rpn[0], Pred_rpn[1], train_img_set, self.loader.class_mapping, use_regr=True, overlap_thresh=0.7, max_boxes=300)

                for index_batch in range(self.args.batch_size):
                    neg_samples = np.where(Y1[index_batch, :, -1] == 1)
                    pos_samples = np.where(Y1[index_batch, :, -1] == 0)

                    if len(neg_samples)>0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []
                    if len(pos_samples)>0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []
                    #选择样本点， 每个batch凑成32个样本进行训练
                    if len(pos_samples) < self.args.num_rois //2 :
                        seleted_pos_samples = pos_samples.tolist()
                    else:
                        seleted_pos_samples = np.random.choice(pos_samples, self.args.num_rois//2, replace=False).tolist()
                    try:
                        seleted_neg_samples = np.random.choice(neg_samples, self.args.num_rois - len(seleted_pos_samples),
                                                               replace=False).tolist()
                    except:
                        seleted_neg_samples = np.random.choice(neg_samples, self.args.num_rois - len(seleted_pos_samples),
                                                               replace=True).tolist()
                    sel_samples = seleted_pos_samples + seleted_neg_samples

                    # all_X = np.expand_dims(X[index_batch, :, :, :], axis=0)
                    # all_x2 = np.expand_dims(X2[index_batch, sel_samples, :], axis=0)
                    # all_y1 = np.expand_dims(Y1[index_batch, sel_samples, :], axis=0)
                    # all_y2 = np.expand_dims(Y2[index_batch, sel_samples, :], axis=0)
                    # loss_class = self.network.classifier_model.train_on_batch([all_X, all_x2], [all_y1, all_y2])

                    loss_class = self.network.classifier_model.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[num_every_current, 0] = loss_rpn[1]
                losses[num_every_current, 1] = loss_rpn[2]
                losses[num_every_current, 2] = loss_class[1]
                losses[num_every_current, 3] = loss_class[2]
                losses[num_every_current, 4] = loss_class[3]

                num_every_current += 1

                progbar.update(num_every_current, [('rpn_cls', np.mean(losses[:num_every_current, 0])),
                                                   ('rpn_regr', np.mean(losses[:num_every_current, 1])),
                                                   ('detector_cls', np.mean(losses[:num_every_current, 2])),
                                                   ('detector_regr', np.mean(losses[:num_every_current, 3])),
                                                   ])


            print('Elapse time:{}'.format(time.time() - start_time))
            current_loss = np.mean(losses[:, 0]) + np.mean(losses[:, 1]) + np.mean(losses[:, 2]) + np.mean(losses[:, 3])
            print('total loss = {}'.format(current_loss))
            if current_loss < self.best_loss:
                print('save model.....')
                self.best_loss = current_loss
                best_name = os.path.basename('./best_' + self.args.model_save_path)
                self.network.model_all.save_weights(best_name)
            self.network.model_all.save_weights(self.args.model_save_path)
        except Exception as e:
            print('Exception:{}'.format(e))

    def test(self):
        print('Testing phase :')
        temp_test_imgs = self.loader.test_imgs

        for file_index in temp_test_imgs:
            img = cv2.imread(file_index['filepath'])
            X, org_size = dt_gen.generate_test_data(self.args, file_index)
            Pred_rpn = self.network.rpn_model.predict_on_batch(X)

            R = roi_helper.rpn_to_roi(self.args, Pred_rpn[0], Pred_rpn[1], overlap_thresh=0.7 )

            #convert from (x1, y1, x2, y2) to (x, y, w, h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            #apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0] // self.args.num_rois + 1):
                ROIs = np.expand_dims(R[self.args.num_rois * jk : self.args.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                #last one
                if jk == R.shape[0] // self.args.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], self.args.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = self.network.classifier_model.predict([X, ROIs])

                for ii in range(P_cls.shape[1]):
                    # 阈值较小 背景类的框 直接舍弃
                    if np.max(P_cls[0, ii, :]) < 0.7 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    a = np.argmax(P_cls[0, ii, :])
                    cls_name = list(self.loader.class_mapping.keys())[list(self.loader.class_mapping.values()).index(a)]


                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]
                    cls_num = np.argmax(P_cls[0, ii, :])


                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self.args.classifier_regr_std[0]
                    ty /= self.args.classifier_regr_std[1]
                    tw /= self.args.classifier_regr_std[2]
                    th /= self.args.classifier_regr_std[3]
                    x, y, w, h = roi_helper.apply_regr(x, y, w, h, tx, ty, tw, th)

                    bboxes[cls_name].append([self.args.rpn_stride*x, self.args.rpn_stride*y, self.args.rpn_stride*(x+w), self.args.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))
            #单独汇总所以的检测结果, 不再以类别的形式汇总
            temp_bbox = []
            temp_prob = []
            pick_key = []
            for key in bboxes:
                temp_bbox += bboxes[key]
                temp_prob += probs[key]
                for asd in range(np.array(bboxes[key]).shape[0]):
                    pick_key.append(key)
            #非极大值抑制
            if len(temp_bbox) == 0:
                print(os.path.basename(file_index['filepath']))
                print('None')
                continue
            new_boxes, new_probs, pick = roi_helper.non_max_suppression_fast(np.array(temp_bbox), np.array(temp_prob), overlap_thresh=0.1)

            all_det=[]
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                key = pick_key[pick[jk]]
                (real_x1, real_y1, real_x2, real_y2) = roi_helper.get_real_coordinates(self.args, x1, y1, x2, y2, org_size)
                #画框
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2)
                textLabel = '{}'.format(key)
                all_det.append((key, 100*new_probs[jk]))
                (retval, baseline) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                textOrg = (real_x1, real_y1)
                #画标签
                cv2.rectangle(img, (textOrg[0] - 1, textOrg[1]+baseline -1), (textOrg[0]+retval[0]+1, textOrg[1]-retval[1]-1),(0,0,0),2)
                cv2.rectangle(img, (textOrg[0] - 1, textOrg[1]+baseline -1), (textOrg[0]+retval[0]+1, textOrg[1]-retval[1]-1),(255,255,255),-1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1)

            print(os.path.basename(file_index['filepath']))
            print(all_det)
            cv2.imwrite('./results_imgs/{}'.format(os.path.basename(file_index['filepath'])), img)


    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            if self.epoch_current< self.args.epochs:
                return False
            else:
                return True





