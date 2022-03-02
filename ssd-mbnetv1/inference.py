import cv2, os, sys
import numpy as np
from onnx import numpy_helper
import onnx
import os, statistics
from PIL import Image, ImageDraw, ImageColor
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt
import onnxruntime as rt
from scipy import special
import colorsys
import random
import time
import glob
import argparse
import json
import pprint
import pandas as pd
from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_detection(draw, d, c):
    """Draw box and label for 1 detection."""
    coco_classes = read_class_names("coco.names")
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = coco_classes[c]
    label_size = draw.textsize(label)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    draw.text(text_origin, label, fill=color)  # , font=font)

def categoryID2name(coco_annotation, query_id):
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    # print(" - Category ID -> Category Name:")
    # print(f" - Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}")
    return query_name

def bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def main():
    # ======= Command-line Arguments =======
    parser = argparse.ArgumentParser(description='Run the model on given imgae dataset for object detection. acc and FPS are reported.')
    parser.add_argument('-m', '--model', metavar='MODEL', required=True,
                        help='path of the model')
    parser.add_argument('-i', '--input', metavar='INPUT', required=True,
                        help='path to the input image folder')
    parser.add_argument('-a', '--annotation', metavar='ANNOTATION', required=True,
                        help='path to the annotation json file')
    parser.add_argument('-s', '--stop', metavar='STOP', type=int, default=np.inf,
                        help='set a breaking point to stop early for testing')
    parser.add_argument('-g', '--gpu', dest='use GPU', action='store_false',
                        help='use GPU instead of CPU. Deafult as CPU only')
    parser.add_argument('-q', '--quantized', dest='quantized', action='store_true',
                        help='set to decide whether we used a quantized model or not')
    parser.add_argument('--save', metavar='SAVE',
                        help='path to save the visual output')
    args = parser.parse_args()

    # ======= Inference =======
    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # rt.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])

    # Load gdourntruth annotations in to a dictionary
    # https://leimao.github.io/blog/Inspecting-COCO-Dataset-Using-COCO-API/
    coco_annotation_file_path = args.annotation
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs and # All categories..
    cat_ids = coco_annotation.getCatIds()
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]

    img_ids = coco_annotation.getImgIds()[-args.stop:]
    print(f"Number of Images to be processed: {len(img_ids)}\n")

    sess = rt.InferenceSession(args.model)
    outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]

    acc_list, conf_list, FPS = [], [], []
    for img_id in tqdm(img_ids):
        # ======= Read the original image =======
        img_info = coco_annotation.loadImgs([img_id])[0]
        img_path = os.path.join(args.input, img_info['file_name'])
        img = Image.open(img_path)
        img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
        img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
        print(img_data.shape)

        # ======= Processing =======
        annIds = coco_annotation.getAnnIds(imgIds=[img_id],  iscrowd=None)
        anns = coco_annotation.loadAnns(annIds)
        start = time.time()
        result = sess.run(outputs, {"image_tensor:0": img_data})
        num_detections, detection_boxes, detection_scores, detection_classes = result
        end = time.time()

        print(detection_boxes)
        # ======= Scores & Output =======
        # output_im = Image.fromarray(image)
        # gth_im = cv2.imread(img_path)
        # detect_count = 0
        # for ann in anns: # gth
        #     query_id = ann['category_id']
        #     query_name = categoryID2name(coco_annotation, query_id)
        #     [x,y,w,h] = [int(i) for i in ann['bbox']]
        #     boo = False
        #     for i, bbox in enumerate(bboxes): # output
        #         bbox_gth = np.array([x, y, x+w, y+h,])
        #         if bboxes_iou(bbox_gth, bbox[:4]) > 0.5 and query_name == names[i]:
        #             boo = True
        #     detect_count += 1 if boo == True else 0
        #
        # print(f' - detected: {detect_count}')

        # draw bbox on the image
        batch_size = num_detections.shape[0]
        draw = ImageDraw.Draw(img)
        for batch in range(0, batch_size):
            for detection in range(0, int(num_detections[batch])):
                c = detection_classes[batch][detection]
                d = detection_boxes[batch][detection]
                draw_detection(draw, d, c)

        plt.figure(figsize=(80, 40))
        plt.axis('off')
        plt.imsave('output.jpg', img)

if __name__ == '__main__':
    main()
