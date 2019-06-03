# -*- coding: utf-8 -*-
import data_loaders.pascal_voc
import data_loaders.common_dir
import pickle


class DataLoader():
    def __init__(self, args):
        self.args = args
        if not self.args.test_only:
            print('loading training dataset')
            self.train_imgs, self.test_imgs,self.class_count, self.class_mapping = pascal_voc.get_data(self.args.VOC_path)
            if 'bg' not in self.class_count:
                self.class_count['bg'] = 0
                self.class_mapping['bg'] = len(self.class_mapping)
            self.args.class_len = len(self.class_count)

            # #封装data的参数
            with open(self.args.config_filename, 'wb') as config_f:
                pickle.dump(self.class_mapping, config_f)
                print('Config file has been written')

        else:
            print('loading testing dataset')
            self.test_imgs = common_dir.get_data(self.args.test_img_path)
            with open(self.args.config_filename, 'rb') as f_in:
                self.class_mapping = pickle.load(f_in)
                print('load Config file')
            self.train_imgs = None
            self.args.class_len = len(self.class_mapping)





