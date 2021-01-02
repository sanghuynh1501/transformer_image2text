import os
from os.path import join
import json
import random
import numpy as np
import cv2
from tqdm import tqdm

from data_hdf5 import HDF5DatasetWriter

letters = ['pad', '<', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P',
           'T', 'X', 'Y', '>']

pad_idx = letters.index('pad')
max_data_len = 5


def text_to_labels(text):
    return [1] + list(map(lambda x: letters.index(x), text)) + [len(letters) - 1]


def build_data(ann_dirpath, img_dirpath):
    img_list = []
    des_list = []
    for filename in os.listdir(img_dirpath):
        name, ext = os.path.splitext(filename)
        if ext in ['.png', '.jpg']:
            img_filepath = join(img_dirpath, filename)
            json_filepath = join(ann_dirpath, name + '.png.json')
            ann = json.load(open(json_filepath, 'r'))
            description = ann['description']
            img_list.append(img_filepath)
            des_list.append(description)

    return img_list, des_list


def merge_data(data_length, total_sample):
    count = 0
    indexes = np.arange(0, data_length, 1).tolist()
    data = []
    while count < total_sample:
        k = random.randrange(6)
        sample = random.choices(indexes, k=k)
        if len(sample) > 0:
            data.append(sample)
            count += 1

    return data


image_hash = {}


def cocat_image(image_list, des_list, img_w, img_h):
    images = np.array([])
    des = ''
    for idx, link in enumerate(image_list):
        img = cv2.imread(link)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_w, img_h))
        if len(img.shape) == 2:
            if len(images) == 0:
                images = img
            else:
                images = np.concatenate([images, img], 1)
            des += des_list[idx]
    return images.T, des


def padding_image(image, max_len):
    while len(image) < max_len:
        image = np.concatenate([image, np.ones((1, image.shape[1])) * pad_idx], 0)
    return image


def padding_text(text, max_len):
    while len(text) < max_len:
        text = np.concatenate([text, np.ones((1,)) * pad_idx], 0)
    return text


def data_process(data_path, ann_link, img_link, data_len, img_w, img_h):
    img_list, des_list = build_data(ann_link, img_link)
    data = merge_data(len(img_list), data_len)

    dump_data = HDF5DatasetWriter((data_len, max_data_len * img_w, img_h), (data_len, max_data_len * 8 + 2), data_path)

    with tqdm(total=len(data)) as pbar:
        for sample in data:
            image, des = cocat_image([img_list[idx] for idx in sample], [des_list[idx] for idx in sample], img_w, img_h)
            image_len = np.array([[image.shape[0]]])
            des_len = np.array([[len(des) + 2]])
            image = padding_image(image, max_data_len * img_w)
            image = np.expand_dims(image, 0)
            des = text_to_labels(des)
            des = padding_text(des, max_data_len * 8 + 2)
            des = np.expand_dims(des, 0)
            dump_data.add(image, des, image_len, des_len)

            pbar.update(1)


data_process('train.hdf5', 'data/anpr_ocr__train/train/ann', 'data/anpr_ocr__train/train/img', 200000, 64, 32)
data_process('test.hdf5', 'data/anpr_ocr__train/test/ann', 'data/anpr_ocr__train/test/img', 20000, 64, 32)
