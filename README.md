Implementation of SINet
=======
# SINet implementation

Implementation of the paper ["SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder"](https://arxiv.org/pdf/1911.09099.pdf) for portrait segmentation in PyTorch including pretrained weights and training script reaching 95.25 IoU on the EG1800 dataset (trained with same schedule and data as original authors).

# Sample results

<div align="center">
<img src="samples/portrait.jpg" width="100px"/>
<img src="samples/mask.png" width="100px"/>
<img src="samples/result.jpg" width="100px"/>
</div>

# Usage

- place data in "datasets" folder (data available through https://github.com/HYOJINPARK/ExtPortraitSeg)
```
ls -l datasets
EG1800
Nukki
--\
   baidu_V1
   baidu_V2

Install requirements
```
pip install -r requirements.txt
```

Run training
```
python train.py [--skip-encoder] [--use-cuda]
```

# Todo
Fix issue with different input image size than 244
