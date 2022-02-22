#!/bin/sh

python inference.py \
  --input /home/lpmot/Dataset/COCO2017/val2017/ \
  --model model/yolov4.onnx \
  -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
  --stop 100 > output/output.txt

python inference.py \
  --input /home/lpmot/Dataset/COCO2017/val2017/ \
  --model model/yolov4-416.onnx \
  -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
  --stop 100\
  --quantized >> output/output.txt

python inference.py \
  --input /home/lpmot/Dataset/COCO2017/val2017/ \
  --model model/yolov4-416-fp16.onnx \
  -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
  --stop 100\
  --quantized >> output/output.txt

python inference.py \
  --input /home/lpmot/Dataset/COCO2017/val2017/ \
  --model model/yolov4-416-int8.onnx \
  -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
  --stop 100\
  --quantized >> output/output.txt

python inference.py \
  --input /home/lpmot/Dataset/COCO2017/val2017/ \
  --model model/yolov4-416-int16.onnx \
  -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
  --stop 100\
  --quantized >> output/output.txt
