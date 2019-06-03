# -*- coding: utf-8 -*-
import xml.etree.ElementTree as XML
import os


def get_data(input_path):
    #input_path include  Annotations, ImageSets and JPEGImages
    #读XML文件，读取训练集，测试集，图片的文件名，宽高，类型名及其目标框左上角坐标和右下角坐标
    #
    #class_count 全部类型记录, class_mapping 记录了每个类别对应的 标号
    #all_imgs 包含了每张图片的文件名和各类标签信息
    class_count = {}
    class_mapping = {}
    test_imgs = []
    train_imgs = []

    # 1li  0oO Ww cC Zz pP Ss Vv Uu Xx
    # char_set = dict(
    #     [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9),
    #      ('a', 10), ('b', 11), ('c', 12), ('d', 13), ('e', 14), ('f', 15), ('g', 16), ('h', 17), ('j', 18),
    #      ('k', 19), ('m', 20), ('n', 21), ('p', 22), ('q', 23), ('r', 24), ('s', 25), ('t', 26), ('u', 27),
    #      ('v', 28), ('w', 29), ('x', 30), ('y', 31), ('z', 32), ('A', 33), ('B', 34), ('D', 35), ('E', 36),
    #      ('F', 37), ('G', 38), ('H', 39), ('I', 40), ('J', 41), ('K', 42), ('L', 43), ('M', 44), ('N', 45),
    #      ('Q', 46), ('R', 47), ('T', 48), ('Y', 49)]
    # )

    char_set = dict(
        [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
    )


    for idx in char_set:
        class_mapping[idx] = char_set[idx]


    annot_path = os.path.join(input_path, 'Annotations')
    imgs_path = os.path.join(input_path, 'JPEGImages')
    imgsets_path_trainval = os.path.join(input_path, 'ImageSets', 'Main', 'trainval.txt')
    imgsets_path_test = os.path.join(input_path, 'ImageSets', 'Main', 'test.txt')

    trainval_files = []
    test_files = []

    try:
        with open(imgsets_path_trainval) as f:
            for line in f:
                trainval_files.append(line.strip() + '.jpg')
    except Exception as e:
        print(e)

    try:
        with open(imgsets_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
    except Exception as e:
        print(e)

    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    for annot in annots:
        try:
            et = XML.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)


            if len(element_objs) > 0:
                annotation_data = {'filepath': os.path.join(imgs_path, element_filename),
                                   'width': element_width,
                                   'height': element_height,
                                   'bboxes': [] }
            else:
                continue

            for element_obj in element_objs:
                class_name = element_obj.find('name').text

                # 1li  0oO Ww cC Zz pP Ss Vv Uu Xx
                if class_name == 'l' or class_name == 'i':
                    class_name = '1'
                elif class_name == 'o' or class_name == 'O':
                    class_name = '0'
                elif class_name == 'W':
                    class_name = 'w'
                elif class_name == 'C':
                    class_name = 'c'
                elif class_name == 'Z':
                    class_name = 'z'
                elif class_name == 'P':
                    class_name = 'p'
                elif class_name == 'S':
                    class_name = 's'
                elif class_name == 'V':
                    class_name = 'v'
                elif class_name == 'U':
                    class_name = 'u'
                elif class_name == 'X':
                    class_name = 'x'

                if class_name not in class_count:
                    class_count[class_name] = 1
                else:
                    class_count[class_name] += 1

#                if class_name not in class_mapping:
#                    class_mapping[class_name] = char_set[class_name]

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                annotation_data['bboxes'].append({'class': class_name, 'x1': x1, 'y1':y1,
                                                  'x2': x2, 'y2': y2})
            if element_filename in trainval_files:
                annotation_data['imageset'] = 'trainval'
                train_imgs.append(annotation_data)
            elif element_filename in test_files:
                annotation_data['imageset'] = 'test'
                test_imgs.append(annotation_data)
            else:
                annotation_data['imageset'] = 'unknown'


        except Exception as e:
            print(e)
            continue
    return train_imgs, test_imgs, class_count, class_mapping

