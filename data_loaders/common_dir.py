# -*- coding: utf-8 -*-
import os

def get_data(input_path):
    test_imgs = []
    img_count = 0

    for idx, img_name in enumerate(sorted(os.listdir(input_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', 'tiff')):
            continue
        annotation_data = {'filepath': os.path.join(input_path, img_name)}

        test_imgs.append(annotation_data)
        img_count += 1

    print("Total test {} images".format(str(img_count)))
    return test_imgs







