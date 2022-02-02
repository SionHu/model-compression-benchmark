## Legend

1. KD: Knowledge Distillation
2. PR: Pruning
3. QZ: Quantization

## Pruning

|<center>method</center> |<center>advantage</center>|<center>disadvantage</center> |
| --- | --- | --- |
|Normal pruning        |Stable compression rate, needs no fine-tuning. |The compression rate is limited.  |
|Shortcut pruning      |Very high compression rate.  | Fine-tuning is necessary.  |
|Silmming              |Shortcut fusion method => improve the precision of shear planting.| Good way for shortcut pruning|
|Regular pruning       |Designed for hardware deploymnet. # Filters = multiple of 2. No fine tuning. | Compression ratio is traded with regularization. |
|layer pruning         |Resblock -> Basic unit, great for hardware. | Cuts only backbone. |
|layer-channel pruning |Use channel pruning -> use layer pruning = pruning rate high. |Accuracy may be affected. |

<br> 

1. COD          https://github.com/Gedeon-m-gedus/compressed_object_detection
    1. Extended pruning for model zoos.
    2. Research paper, lots of references.


2. AIMET            https://github.com/quic/aimet
    1. Model Zoo
    2. Good Documentation   https://quic.github.io/aimet-pages/releases/1.16.2/api_docs/index.html
    3. Has Channel Pruning

3. Distiller
    1. See Below


4. Tensorflow model optimization    https://github.com/tensorflow/model-optimization


## Knowledge Distillation

<br> <br>

1. KD-lib (KD, PR, QZ)              https://github.com/SforAiDl/KD_Lib
    1. Easy to use 
    2. Good logging 
    3. Still being maintained
    4. Great Documentation: https://kd-lib.readthedocs.io/en/latest/ 

<br> <br>

2. Distiller (KD, PR, QZ, More)     https://github.com/IntelLabs/distiller
    1. 3,000+ stars 
    2. Intel AI Lab 
    3. Great Documentation: https://intellabs.github.io/distiller/ 
    4. Harder to use ❗

<br> <br>

3. Aquivatae (KD)                   https://github.com/marload/aquvitae
    1. easy to use API 
 
<br> <br>

4. torchdistill (KD)                https://github.com/yoshitomo-matsubara/torchdistill
    1. presented at CVPR, ICLR, ECCV, NeurIPS, ICCV 
    2. Many many references 
    3. Google colab examples, good documentation 

<br> <br>

5. MT-DNN (KD)                      https://github.com/microsoft/MT-DNN
    1. Microsoft 
    2. Has references 
    3. I'm not sure how to use this it's a bit too much. ❗



# MISC

Great Resource to study & see some prelimary benchmarks on YoloV5 for mobile:
https://github.com/Syencil/mobile-yolov5-pruning-distillation
(In chineese, use translate)

Specific libraries for model zoo: (Alot of this in chineese idk why?)
https://www.opensourceagenda.com/tags/pruning


papers on a ton of additional libraries for all compression methods:
https://github.com/cedrickchee/awesome-ml-model-compression


easy api for pruning:
https://nathanhubens.github.io/fasterai/pruner.html 
