# Supervised Compression for Resource-constrained Edge Computing Systems

## Input Compression vs. Supervised Compression
![Input compression vs. Feature compression](imgs/input_vs_feature_compression.png) 
---  

## Proposed Supervised Compression
![Proposed training method](imgs/proposed_training_method.png) 

## Citation
[[Preprint](https://arxiv.org/abs/2108.11898)]  
```bibtex
@article{matsubara2021supervised,
  title={Supervised Compression for Resource-constrained Edge Computing Systems},
  author={Matsubara, Yoshitomo and Yang, Ruihan and Levorato, Marco and Mandt, Stephan},
  journal={arXiv preprint arXiv:2108.11898},
  year={2021}
}
```

## Requirements
- Python >= 3.6.9
- pipenv

## 1. Virtual Environment Setup
```
# For Python 3.6 users
pipenv install --python 3.6

# For Python 3.7 users
pipenv install --python 3.7

# For Python 3.8 users
pipenv install --python 3.8
```

## 2. Download Datasets

### 2.1 ImageNet (ILSVRC 2012): Image Classification
As the terms of use do not allow to distribute the URLs, 
you will have to create an account [here](http://image-net.org/download) to get the URLs, 
and replace `${TRAIN_DATASET_URL}` and `${VAL_DATASET_URL}` with them.

```shell
wget ${TRAIN_DATASET_URL} ./
wget ${VAL_DATASET_URL} ./
```

Untar and extract files
```shell
mkdir ~/dataset/ilsvrc2012/{train,val} -p
mv ILSVRC2012_img_train.tar ~/dataset/ilsvrc2012/train/
cd ~/dataset/ilsvrc2012/train/
tar -xvf ILSVRC2012_img_train.tar
mv ILSVRC2012_img_train.tar ../
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xf ../$f)
done
rm -r *.tar

mv ILSVRC2012_img_val.tar ~/dataset/ilsvrc2012/val/
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/val/
tar -xvf ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar ../
sh valprep.sh
mv valprep.sh ../
```


### 2.2 COCO 2017: Object Detection & Semantic Segmentation
Download and unzip the datasets
```shell
mkdir ~/dataset/coco2017/ -p
cd ~/dataset/coco2017/
wget http://images.cocodataset.org/zips/train2017.zip ./
wget http://images.cocodataset.org/zips/val2017.zip ./
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ./
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### 3. Input Compression (IC) Baselines

#### JPEG Codec
```shell
# Image classification
echo 'jpeg quality=100'
pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/jpeg-resnet50.yaml

for quality in $(seq 100 -10 20)
do
  next_quality=$((quality-10))
  echo 'jpeg quality=${next_quality}'
  sed -i "s/jpeg_quality: ${quality}/jpeg_quality: ${next_quality}/" configs/ilsvrc2012/input_compression/jpeg-resnet50.yaml
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/jpeg-resnet50.yaml
done

# Object detection
echo 'jpeg quality=100'
pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/jpeg-retinanet_resnet50_fpn.yaml

for quality in $(seq 100 -10 20)
do
  next_quality=$((quality-10))
  echo 'jpeg quality=${next_quality}'
  sed -i "s/jpeg_quality: ${quality}/jpeg_quality: ${next_quality}/" configs/coco2017/input_compression/jpeg-retinanet_resnet50_fpn.yaml
  pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/jpeg-retinanet_resnet50_fpn.yaml
done

# Semantic segmentation
echo 'jpeg quality=100'
pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/jpeg-deeplabv3_resnet50.yaml

for quality in $(seq 100 -10 20)
do
  next_quality=$((quality-10))
  echo 'jpeg quality=${next_quality}'
  sed -i "s/jpeg_quality: ${quality}/jpeg_quality: ${next_quality}/" configs/coco2017/input_compression/jpeg-deeplabv3_resnet50.yaml
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/jpeg-deeplabv3_resnet50.yaml
done

```

#### WebP Codec
```shell
# Image classification
echo 'webp quality=100'
pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/webp-resnet50.yaml

for quality in $(seq 100 -10 20)
do
  next_quality=$((quality-10))
  echo 'webp quality=${next_quality}'
  sed -i "s/webp_quality: ${quality}/webp_quality: ${next_quality}/" configs/ilsvrc2012/input_compression/webp-resnet50.yaml
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/webp-resnet50.yaml
done

# Object detection
echo 'webp quality=100'
pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/webp-retinanet_resnet50_fpn.yaml

for quality in $(seq 100 -10 20)
do
  next_quality=$((quality-10))
  echo 'webp quality=${next_quality}'
  sed -i "s/webp_quality: ${quality}/webp_quality: ${next_quality}/" configs/coco2017/input_compression/webp-retinanet_resnet50_fpn.yaml
  pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/webp-retinanet_resnet50_fpn.yaml
done

# Semantic segmentation
echo 'webp quality=100'
pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/webp-deeplabv3_resnet50.yaml

for quality in $(seq 100 -10 20)
do
  next_quality=$((quality-10))
  echo 'webp quality=${next_quality}'
  sed -i "s/webp_quality: ${quality}/webp_quality: ${next_quality}/" configs/coco2017/input_compression/webp-deeplabv3_resnet50.yaml
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/webp-deeplabv3_resnet50.yaml
done
```

#### BPG Codec
Install BPG following the instructions [here](https://bellard.org/bpg/).  
If you do not place **bpgenc** and **bpgdec** at '~/manually_installed/libbpg-0.9.8/', 
edit the `encoder_path` and `decoder_path` in `bpg-resnet.yaml` like [this](https://github.com/yoshitomo-matsubara/supervised-compression/blob/main/configs/ilsvrc2012/input_compression/bpg-resnet50.yaml#L19-L20).

```shell
# Image classification
echo 'bpg quality=50'
pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/bpg-resnet50.yaml

for quality in $(seq 50 -5 5)
do
  next_quality=$((quality-5))
  echo 'bpg quality=${next_quality}'
  sed -i "s/bpg_quality: ${quality}/bpg_quality: ${next_quality}/" configs/ilsvrc2012/input_compression/bpg-resnet50.yaml
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/bpg-resnet50.yaml
done

# Object detection
echo 'bpg quality=50'
pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/bpg-retinanet_resnet50_fpn.yaml

for quality in $(seq 50 -5 5)
do
  next_quality=$((quality-5))
  echo 'bpg quality=${next_quality}'
  sed -i "s/bpg_quality: ${quality}/bpg_quality: ${next_quality}/" configs/coco2017/input_compression/bpg-retinanet_resnet50_fpn.yaml
  pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/bpg-retinanet_resnet50_fpn.yaml
done

# Semantic segmentation
echo 'bpg quality=50'
pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/bpg-deeplabv3_resnet50.yaml

for quality in $(seq 50 -5 5)
do
  next_quality=$((quality-5))
  echo 'bpg quality=${next_quality}'
  sed -i "s/bpg_quality: ${quality}/bpg_quality: ${next_quality}/" configs/coco2017/input_compression/bpg-deeplabv3_resnet50.yaml
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/bpg-deeplabv3_resnet50.yaml
done
```

#### Factorized Prior
Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. "Variational image compression with a scale hyperprior"  

Make sure you have downloaded and unzipped the pretrained checkpoints for this model. The checkpoint files should be 
placed at `./resource/ckpt/input_compression/` as specified in the yaml file.

```shell
# Image classification
echo 'beta=0.00015625'
pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/factorized_prior_ae_128ch-resnet50.yaml
prev_beta=0.00015625

for beta in 0.0003125 0.000625 0.00125 0.0025 0.005 0.01 0.02
do
  echo 'beta=${beta}'
  sed -i "s/beta_${prev_beta}/beta_${beta}/" configs/ilsvrc2012/input_compression/factorized_prior_ae_128ch-resnet50.yaml
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/factorized_prior_ae_128ch-resnet50.yaml
  prev_beta=${beta}
done

# Object detection
echo 'beta=0.00015625'
pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/factorized_prior_ae_128ch-retinanet_resnet50_fpn.yaml
prev_beta=0.00015625

for beta in 0.0003125 0.000625 0.00125 0.0025 0.005 0.01 0.02
do
  echo 'beta=${beta}'
  sed -i "s/beta_${prev_beta}/beta_${beta}/" configs/coco2017/input_compression/factorized_prior_ae_128ch-retinanet_resnet50_fpn.yaml
  pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/factorized_prior_ae_128ch-retinanet_resnet50_fpn.yaml
  prev_beta=${beta}
done

# Semantic segmentation
echo 'beta=0.00015625'
pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/factorized_prior_ae_128ch-deeplabv3_resnet50.yaml
prev_beta=0.00015625

for beta in 0.0003125 0.000625 0.00125 0.0025 0.005 0.01 0.02
do
  echo 'beta=${beta}'
  sed -i "s/beta_${prev_beta}/beta_${beta}/" configs/coco2017/input_compression/factorized_prior_ae_128ch-deeplabv3_resnet50.yaml
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/factorized_prior_ae_128ch-deeplabv3_resnet50.yaml
  prev_beta=${beta}
done
```


#### Mean-scale Hyperprior 
David Minnen, Johannes Ballé, and George D Toderici. "Joint autoregressive and hierarchical priors for learned image compression"  

Make sure you have downloaded and unzipped the pretrained checkpoints for this model. The checkpoint files should be 
placed at `./resource/ckpt/input_compression/` as specified in the yaml file.

```shell
# Image classification
echo 'beta=0.00015625'
pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/hierarchical_prior_ae_128ch-resnet50.yaml
prev_beta=0.00015625

for beta in 0.0003125 0.000625 0.00125 0.0025 0.005 0.01 0.02
do
  echo 'beta=${beta}'
  sed -i "s/beta_${prev_beta}/beta_${beta}/" configs/ilsvrc2012/input_compression/hierarchical_prior_ae_128ch-resnet50.yaml
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/input_compression/hierarchical_prior_ae_128ch-resnet50.yaml
  prev_beta=${beta}
done

# Object detection
echo 'beta=0.00015625'
pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/hierarchical_prior_ae_128ch-retinanet_resnet50_fpn.yaml
prev_beta=0.00015625

for beta in 0.0003125 0.000625 0.00125 0.0025 0.005 0.01 0.02
do
  echo 'beta=${beta}'
  sed -i "s/beta_${prev_beta}/beta_${beta}/" configs/coco2017/input_compression/hierarchical_prior_ae_128ch-retinanet_resnet50_fpn.yaml
  pipenv run python object_detection.py -test_only --config configs/coco2017/input_compression/hierarchical_prior_ae_128ch-retinanet_resnet50_fpn.yaml
  prev_beta=${beta}
done

# Semantic segmentation
echo 'beta=0.00015625'
pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/hierarchical_prior_ae_128ch-deeplabv3_resnet50.yaml
prev_beta=0.00015625

for beta in 0.0003125 0.000625 0.00125 0.0025 0.005 0.01 0.02
do
  echo 'beta=${beta}'
  sed -i "s/beta_${prev_beta}/beta_${beta}/" configs/coco2017/input_compression/hierarchical_prior_ae_128ch-deeplabv3_resnet50.yaml
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/input_compression/hierarchical_prior_ae_128ch-deeplabv3_resnet50.yaml
  prev_beta=${beta}
done
```

### 4. Feature Compression (FC) Baselines

#### Channel Reduction and Bottleneck Quantization
Yoshitomo Matsubara, Marco Levorato. ["Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks"](https://github.com/yoshitomo-matsubara/hnd-ghnd-object-detectors)  

Make sure you have downloaded and unzipped the pretrained checkpoints for this model. The checkpoint files should be 
placed at `./resource/ckpt/ilsvrc2012/bq/` and  `./resource/ckpt/coco2017/bq/` as specified in the yaml file.

```shell
# Image classification
for bch in 12 9 6 3 2
do
  echo 'bottleneck channel=${bch}'
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/bq/custom_resnet50-bq${bch}ch_from_resnet50.yaml
done

# Object detection
for bch in 6 3 2 1
do
  echo 'bottleneck channel=${bch}'
  pipenv run python object_detection.py -test_only --config configs/coco2017/bq/custom_retinanet_resnet50_fpn-bq${bch}ch_from_retinanet_resnet50_fpn.yaml
done

# Semantic segmentation
for bch in 9 6 3 2 1
do
  echo 'bottleneck channel=${bch}'
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/bq/custom_deeplabv3_resnet50_bq${bch}ch_from_deeplabv3_resnet50.yaml
done
```

#### End-to-end training
Saurabh Singh, Sami Abu-El-Haija, Nick Johnston, Johannes Ballé, Abhinav Shrivastava, George Toderici. "End-to-End Learning of Compressible Features"  

Make sure you have downloaded and unzipped the pretrained checkpoints for this model. The checkpoint files should be 
placed at `./resource/ckpt/ilsvrc2012/singh_et_al/` and `./resource/ckpt/coco2017/shared_singh_et_al/` as specified in the yaml file.

```shell
# Image classification
for beta in 5.0e-11 1.0e-10 2.0e-10 4.0e-10 8.0e-10 1.6e-9 6.4e-9 1.28e-8
do
  echo 'beta=${beta}'
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/singh_et_al/bottleneck_resnet50-b24ch_igdn-beta${beta}_from_resnet50.yaml
done

# Object detection
for beta in 8.0e-10 1.6e-9 6.4e-9 1.28e-8
do
  echo 'beta=${beta}'
  pipenv run python object_detection.py -test_only --config configs/coco2017/shared_singh_et_al/retinanet_bottleneck_resnet50-b24ch_igdn-beta${beta}_fpn.yaml
done

# Semantic segmentation
for beta in 8.0e-10 1.6e-9 6.4e-9 1.28e-8
do
  echo 'beta=${beta}'
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/shared_singh_et_al/deeplabv3_bottleneck_resnet50-b24ch_igdn-beta${beta}.yaml
done
```

### 5. Our supervised compression
Make sure you have downloaded and unzipped the pretrained checkpoints for this model. The checkpoint files should be 
placed at `./resource/ckpt/ilsvrc2012/pre_ghnd-kd/` and `./resource/ckpt/coco2017/shared_encoder/` as specified in the yaml file.

```shell
# Image classification
for beta in 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 
do
  echo 'beta=${beta}'
  pipenv run python image_classification.py -test_only --config configs/ilsvrc2012/pre_ghnd-kd/bottleneck_resnet50-b24ch_igdn-beta${beta}_from_resnet50.yaml
done

# Object detection
for beta in 0.04 0.08 0.16 0.32 0.64 1.28 2.56 
do
  echo 'beta=${beta}'
  pipenv run python object_detection.py -test_only --config configs/coco2017/shared_encoder/retinanet_bottleneck_resnet50-b24ch_igdn-beta${beta}_fpn_from_retinanet_resnet50_fpn.yaml
done

# Semantic segmentation
for beta in 0.04 0.08 0.16 0.32 0.64 1.28 2.56 
do
  echo 'beta=${beta}'
  pipenv run python semantic_segmentation.py -test_only --config configs/coco2017/shared_encoder/deeplabv3_bottleneck_resnet50-b24ch_igdn-beta${beta}_from_deeplabv3_resnet50.yaml
done
```
