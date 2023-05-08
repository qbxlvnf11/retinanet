Description
=============

#### - The class imbalance problem of object detector
  - Arising it when there is a significant difference between the number of positive samples (foreground objects) and negative samples (background) in the dataset
  - This can lead to the model being biased towards the majority class and perform poorly on the minority class
  - Class imbalance problem in one-stage detector is more severe than in two-stage detector because it performs dense sampling method that densely traverses and samples the entire image without region proposal process

#### - Focus loss of RetinaNet
  - Loss function to deal with class imbalance problem in one-stage
  - The form of adding a dynamic scaling factor that changes according to the class to the cross entropy loss
  - Automatically down-weight the contribution of easy examples during learning, and increase the weight on hard examples
  
#### - Architecture of RetinaNet

<img src="https://user-images.githubusercontent.com/52263269/236881993-859cb50c-2349-45b3-9db1-2dae1151e05c.png" width="60%"></img> 

Contents
=============

#### - Modify RetinaNet code in [pytorch-retinanet official repository](https://github.com/yhenon/pytorch-retinanet) to build RetinaNet object detector optimized to human detection (fine-tuning the model using CrowdHuman dataset)

#### - RetinaNet Train/Fine-tune/Validate/Inference/Visualization

  - Inference results of CrowdHuman (left: gt, right: predicted)
  
  <img src="https://user-images.githubusercontent.com/52263269/236888024-a4486f83-8b5a-4159-83ba-ef949e337fb4.jpg" width="45%"></img> 
  <img src="https://user-images.githubusercontent.com/52263269/236888079-ad5bc26f-a146-4c4d-89e6-a51e62fc963a.jpg" width="45%"></img>

  <img src="https://user-images.githubusercontent.com/52263269/236888241-0dfc0470-394e-4c6b-85dc-68c702a49e25.jpg" width="45%"></img> 
  <img src="https://user-images.githubusercontent.com/52263269/236888258-284726d4-0611-4807-aed1-ed9103c4b64d.jpg" width="45%"></img>

Structures of Project Folders
=============

```
${ROOT}
            |   |-- train.py
            |   |-- csv_validation.py
            |   |-- build_annotations_file.py
            |   |-- ...
            |   |-- class_names
            |   |   |   |-- coco_names_with_head.txt
            |   |   |   |-- coco_names_with_head.csv (Build csv files of class mappings by running python build_class_mapping_file.py)
            |   |-- weights
            |   |   |   |-- coco_resnet_50_map_0_335_state_dict.pt
            |   |-- data
            |   |   |   |-- CrowdHuman
            |   |   |   |   |   |-- CrowdHuman_train01
            |   |   |   |   |   |-- CrowdHuman_train02
            |   |   |   |   |   |-- CrowdHuman_train03
            |   |   |   |   |   |-- CrowdHuman_val
            |   |   |   |   |   |-- CrowdHuman_test
            |   |   |   |   |   |-- annotation_train.odgt
            |   |   |   |   |   |-- annotation_val.odgt
            |   |   |   |   |   |-- train_annotations.csv (Build csv files of annotations by running python build_annotations_file.py)
            |   |   |   |   |   |-- valid_annotations.csv (Build csv files of annotations by running python build_annotations_file.py)
            |   |   |   |-- COCO2017
            |   |   |   |   |   |-- images
            |   |   |   |   |   |-- labels
            |   |   |   |   |   |-- train2017.txt
            |   |   |   |   |   |-- val2017.txt
            |   |   |   |   |   |-- test-dev2017.txt
```


Build Custom CrowdHuman CSV Dataset
=============

#### - Downlaod CrowdHuman Dataset

https://www.crowdhuman.org/

https://www.crowdhuman.org/download.html

#### - Build csv files of class mappings

```
python build_class_mapping_file.py --class_names_txt ./class_names/coco_names_with_head.txt --class_names_csv ./class_names/coco_names_with_head.csv
```

#### - Build csv files of annotations

```
python build_annotations_file.py --dataset crowd_human
```


Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/retinanet_env
```

#### - Run docker environment

```
nvidia-docker run -it --gpus all --name retinanet_env --shm-size=64G -p 8844:8844 -e GRANT_SUDO=yes --user root -v {retinanet_folder}:/workspace/retinanet -w /workspace/retinanet qbxlvnf11docker/retinanet_env bash
```


How to use
=============

#### - Train RetinaNet
  - Train COCO

  ```
  python train.py \
    --dataset coco \
    --coco_path ./data/COCO2017 \
    --depth {18, 34, 50, 101, 152}
  ```

  - Train CrowdHuman
  ```
  python train.py \
    --dataset csv \
    --csv_classes ./class_names/coco_names_with_head.csv \
    --csv_train ./data/CrowdHuman/train_annotations.csv \
    --csv_val ./data/CrowdHuman/valid_annotations.csv \
    --depth {18, 34, 50, 101, 152}
  ```

  - Fine-tune CrowdHuman
  ```
  python train.py \
    --dataset csv \
    --csv_classes ./class_names/coco_names_with_head.csv \
    --csv_train ./data/CrowdHuman/train_annotations.csv \
    --csv_val ./data/CrowdHuman/valid_annotations.csv \
    --start_epoch 0 \
    --depth 50 \
    --model_path ./weights/coco_resnet_50_map_0_335_state_dict.pt
  ```

#### - Validation (mAP)
  - Valid COCO

  ```
  python coco_validation.py \
    --coco_path ./data/COCO2017 \
    --model_path ./weights/coco_resnet_50_map_0_335_state_dict.pt
  ```

  - Valid CrowdHuman
  ```
  python csv_validation.py \
    --csv_classes ./class_names/coco_names_with_head.csv \
    --csv_val ./data/CrowdHuman/valid_annotations.csv \
    --model_path {pretrained_weights_path}
  ```
  
#### - Visualization of inference results
  - Visualization COCO

  ```
  python visualize.py \
    --dataset coco \
    --coco_path ./data/COCO2017 \
    --model ./weights/coco_resnet_50_map_0_335_state_dict.pt
  ```

  - Visualization CrowdHuman
  ```
  python visualize.py \
    --dataset csv \
    --csv_classes ./class_names/coco_names_with_head.csv \
    --csv_val ./data/CrowdHuman/valid_annotations.csv \
    --model_path {pretrained_weights_path}
  ```
  

Download Weights
=============

#### - Download COCO pretrained weights of RetinaNet (coco_resnet_50_map_0_335_state_dict.pt)

https://github.com/yhenon/pytorch-retinanet


Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com

