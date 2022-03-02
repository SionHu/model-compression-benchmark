import cv2
import numpy as np
from onnx import numpy_helper
import onnx
import onnx_tensorrt.backend as backend
import os, statistics
from PIL import Image
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt
# import onnxruntime as rt
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

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes

def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)

def postprocess_bbbox(quantized, pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    '''define anchor boxes'''

    # if used quantized model, we actually dont need to do anything
    if quantized:
        pred_bbox = np.concatenate((pred_bbox[0], pred_bbox[1]), axis=2)
        x, y, z = pred_bbox.shape
        pred_bbox = np.reshape(pred_bbox, (y, z))
        return pred_bbox

    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(float)

        pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox

def postprocess_boxes(quantized, pred_bbox, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4] if not quantized else 1
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

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

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bbox(quantized, image, bboxes, classes=read_class_names("coco.names"), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    returrn image, mean_prob, names
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    probability, names = [], []

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        probability.append(score)
        cls_name = classes[class_ind] if not quantized else classes[class_ind+1]
        names.append(cls_name)

        if show_label:
            bbox_mess = '%s: %.2f' % (cls_name, score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
    mean_prob = statistics.mean(probability) if len(probability) != 0 else 0
    return image, mean_prob, names


def categoryID2name(coco_annotation, query_id):
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    # print(" - Category ID -> Category Name:")
    # print(f" - Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}")
    return query_name

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
    # print(f"Number of Unique Categories: {len(cat_ids)}")
    # print(f"Category IDs: {cat_ids}") # The IDs are not necessarily consecutive.
    # print(f"Categories Names: {cat_names}")

    # # Category ID -> Category Name.
    # query_id = cat_ids[0]
    # query_annotation = coco_annotation.loadCats([query_id])[0]
    # query_name = query_annotation["name"]
    # query_supercategory = query_annotation["supercategory"]
    # print("Category ID -> Category Name:")
    # print(
    #     f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    # )
    # # Category Name -> Category ID.
    # query_name = cat_names[2]
    # query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    # print("Category Name -> ID:")
    # print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds()[-args.stop:]
    print(f"Number of Images to be processed: {len(img_ids)}\n")

    ANCHORS = "./yolov4_anchors.txt"
    STRIDES = [8, 16, 32]
    XYSCALE = [1.2, 1.1, 1.05]

    ANCHORS = get_anchors(ANCHORS)
    STRIDES = np.array(STRIDES)

    # rt.get_device()
    # sess = rt.InferenceSession(args.model, providers=['CUDAExecutionProvider'])
    model = onnx.load(args.model)
    sess = backend.prepare(model, device='CUDA:0')
    input_size = 416

    acc_list, conf_list, FPS = [], [], []
    for img_id in tqdm(img_ids):

        # ======= Read the original image =======
        img_info = coco_annotation.loadImgs([img_id])[0]
        img_path = os.path.join(args.input, img_info['file_name'])
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # print(" - Preprocessed image shape:",image_data.shape) # shape of the preprocessed input
        # imsave("sample.jpg", np.asarray(original_image))

        # ======= Processing =======
        annIds = coco_annotation.getAnnIds(imgIds=[img_id],  iscrowd=None)
        anns = coco_annotation.loadAnns(annIds)
        start = time.time()
        outputs = sess.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        input_name = sess.get_inputs()[0].name
        detections = sess.run(output_names, {input_name: image_data})
        end = time.time()
        # print(" - Output shape:", list(map(lambda detection: detection.shape, detections)))

        # ======= Start Post-processing =======
        pred_bbox = postprocess_bbbox(args.quantized, detections, ANCHORS, STRIDES, XYSCALE)
        bboxes = postprocess_boxes(args.quantized, pred_bbox, original_image_size, input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method='nms')
        image, conf, names = draw_bbox(args.quantized, original_image, bboxes)

        # ======= Scores & Output =======
        output_im = Image.fromarray(image)
        gth_im = cv2.imread(img_path)
        detect_count = 0
        for ann in anns: # gth
            query_id = ann['category_id']
            query_name = categoryID2name(coco_annotation, query_id)
            [x,y,w,h] = [int(i) for i in ann['bbox']]
            boo = False
            for i, bbox in enumerate(bboxes): # output
                bbox_gth = np.array([x, y, x+w, y+h,])
                if bboxes_iou(bbox_gth, bbox[:4]) > 0.5 and query_name == names[i]:
                    boo = True
            detect_count += 1 if boo == True else 0

            # Draw the boxes
            cv2.rectangle(gth_im, (x,y), (x+w, y+h), (0, 255, 0), 2)
            gth_im = cv2.putText(gth_im, query_name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        # print(f' = Found {detect_count} matches. Total {len(anns)}')


        if args.save:
            final = np.concatenate((np.asarray(output_im), np.asarray(gth_im)), axis = 0)
            cv2.imwrite(f'{args.save}/{img_id}.jpg', final)

        conf = round(conf, 6)
        fps = round(1/(end-start), 6)
        acc = round(detect_count / len(anns), 6)
        conf_list.append(conf)
        FPS.append(fps)
        acc_list.append(acc)

        # print(f' - Processing {img_path}')
        # print(' - mConf: {:.4f}; FPS: {:.4f}; acc: {:.4f}'.format(conf, fps, acc))

    print(f' - model: {args.model.split("/")[-1]}, mConfidence: {statistics.mean(conf_list)}, mFPS: {statistics.mean(FPS)}, mAcc: {statistics.mean(acc_list)}\n')

if __name__ == '__main__':
    main()
