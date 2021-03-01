#!/bin/bash -eu
# Share ResNet's params
pipenv run python ckpt_editor.py --source resnet50 --target retinanet_resnet50_fpn --mode overwrite --target_prefix backbone.body. --output edited_ckpt/retinanet_resnet50_fpn-ilsvrc2012.pt
pipenv run python ckpt_editor.py --source resnet50 --target keypointrcnn_resnet50_fpn --mode overwrite --target_prefix backbone.body. --output edited_ckpt/keypointrcnn_resnet50_fpn-ilsvrc2012.pt
pipenv run python ckpt_editor.py --source resnet50 --target deeplabv3_resnet50 --mode overwrite --target_prefix backbone. --output edited_ckpt/deeplabv3_resnet50-ilsvrc2012.pt

# Share params of ResNet in RetinaNet
pipenv run python ckpt_editor.py --source retinanet_resnet50_fpn --target resnet50 --mode overwrite_w_prefix --source_prefix backbone.body. --output edited_ckpt/resnet50-retinanet_resnet50_fpn.pt
pipenv run python ckpt_editor.py --source retinanet_resnet50_fpn --target keypointrcnn_resnet50_fpn --mode overwrite_w_prefix --source_prefix backbone.body. --target_prefix backbone.body. --output edited_ckpt/keypointrcnn_resnet50_fpn-retinanet_resnet50_fpn.pt
pipenv run python ckpt_editor.py --source retinanet_resnet50_fpn --target deeplabv3_resnet50 --mode overwrite_w_prefix --source_prefix backbone.body. --target_prefix backbone. --output edited_ckpt/deeplabv3_resnet50-retinanet_resnet50_fpn.pt

# Share params of ResNet in Keypoint R-CNN
pipenv run python ckpt_editor.py --source keypointrcnn_resnet50_fpn --target resnet50 --mode overwrite_w_prefix --source_prefix backbone.body. --output edited_ckpt/resnet50-keypointrcnn_resnet50_fpn.pt
pipenv run python ckpt_editor.py --source keypointrcnn_resnet50_fpn --target retinanet_resnet50_fpn --mode overwrite_w_prefix --source_prefix backbone.body. --target_prefix backbone.body. --output edited_ckpt/retinanet_resnet50_fpn-keypointrcnn_resnet50_fpn.pt
pipenv run python ckpt_editor.py --source keypointrcnn_resnet50_fpn --target deeplabv3_resnet50 --mode overwrite_w_prefix --source_prefix backbone.body. --target_prefix backbone. --output edited_ckpt/deeplabv3_resnet50-keypointrcnn_resnet50_fpn.pt

# Share params of ResNet in DeepLabv3
pipenv run python ckpt_editor.py --source deeplabv3_resnet50 --target resnet50 --mode overwrite_w_prefix --source_prefix backbone. --output edited_ckpt/resnet50-deeplabv3_resnet50.pt
pipenv run python ckpt_editor.py --source deeplabv3_resnet50 --target retinanet_resnet50_fpn --mode overwrite_w_prefix --source_prefix backbone. --target_prefix backbone.body. --output edited_ckpt/retinanet_resnet50_fpn-deeplabv3_resnet50.pt
pipenv run python ckpt_editor.py --source deeplabv3_resnet50 --target keypointrcnn_resnet50_fpn --mode overwrite_w_prefix --source_prefix backbone. --target_prefix backbone.body. --output edited_ckpt/keypointrcnn_resnet50_fpn-deeplabv3_resnet50.pt
