# deecamp20_3_1_lidar_detection
deecamp20赛题3-1。自动驾驶场景128线点云目标检测，个人实验记录

具体工程地址： [deecamp20-3-1-lidar_det_pcdet](https://github.com/LittleYuanzi/deecamp20-3-1-lidar_det_pcdet)

## 简介
deecamp20赛题3-1，128线下的自动驾驶场景的纯点云目标检测。环境：

 - Linux (tested on Ubuntu 14.04/16.04)
 - Python 3.6+
 - PyTorch 1.1 or higher
 - CUDA 9.0 or higher
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200624171203829.png?x-os-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDgwNTM5Mg==,size_16,color_FFFFFF,t_70)
## 实验对比
 - 官方Baseline
 
|model| 备注|FPS|MAP|
|--|--|--|--|
| second（16epoch） | Det3D版本，<br>采用multi-head和apex加速|6.67|Evaluation official:car AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:70.63<br>3d   AP:68.38<br>aos  AP:0.00<br>car AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:79.27<br>3d   AP:79.14<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:37.04<br>3d   AP:25.94<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:51.19<br>3d   AP:41.73<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:42.09<br>3d   AP:31.61<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:52.19<br>3d   AP:52.07<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:61.12<br>3d   AP:60.53<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:61.59<br>3d   AP:61.58<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:38.22<br>3d   AP:36.45<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:50.67<br>3d   AP:50.63<br>aos  AP:0.00|

官方给出的base在FPS也是需要改进才能做到>10FPS的。
 - 没有考虑速度的情形下的实验
 
|model| 备注|FPS|MAP|
|--|--|--|--|
|second(16epoch)|openpcdet版本,<br>没有multihead，无apex加速,<br>采用gt_数据增广|5|Car AP@0.70, 0.70, 0.70:<br>bev  AP:58.5172<br>3d   AP:19.5790<br>aos  AP:0.00<br>Car AP@0.70, 0.50, 0.50:<br>bev  AP:67.5123<br>3d   AP:62.4208<br>aos  AP:0.00<br>Truck AP@0.50, 0.50, 0.50:<br>bev  AP:12.6986<br>3d   AP:9.3443<br>aos  AP:0.00<br>Truck AP@0.50, 0.25, 0.25:<br>bev  AP:23.7338<br>3d   AP:20.7437<br>aos  AP:0.00<br>Tricar AP@0.50, 0.50, 0.50:<br>bev  AP:16.6359<br>3d   AP:8.7095<br>aos  AP:0.00<br>Tricar AP@0.50, 0.25, 0.25:<br>bev  AP:20.7536<br>3d   AP:19.3297<br>aos  AP:0.00<br>Cyclist AP@0.70, 0.70, 0.70:<br>bev  AP:7.4312<br>3d   AP:2.2727<br>aos  AP:0.00<br>Cyclist AP@0.70, 0.50, 0.50:<br>bev  AP:34.3507<br>3d   AP:23.6472<br>aos  AP:0.00<br>Pedestrian AP@0.50, 0.50, 0.50:<br>bev  AP:9.0909<br>3d   AP:9.0909<br>aos  AP:0.00<br>Pedestrian AP@0.50, 0.25, 0.25:<br>bev  AP:10.2772<br>3d   AP:10.2405<br>aos  AP:0.00|
|PV-RCNN1(6epoch)|open-pcdet，two-stage，无apex加速，<br>采用gt_数据增广|3.37|Evaluation official: car AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:77.19<br>3d   AP:69.01<br>aos  AP:0.00<br>car AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:78.82<br>3d   AP:78.46<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:56.95<br>3d   AP:46.56<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:64.10<br>3d   AP:58.87<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:31.15<br>3d   AP:21.79<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br><br>bev  AP:38.87<br>3d   AP:35.25<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:61.22<br>3d   AP:55.64<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:67.46<br>3d   AP:66.95<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:35.08<br>3d   AP:30.14<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:44.98<br>3d   AP:41.03<br>aos  AP:0.00|

以上是在没有考虑速度下个人做的一些实验，其中PVRCNN是目前在KITTI上目标检测效果最好的文章，发表在的CVPR20，同时也获得了waymo CVPR20挑战赛的两个冠军，这个实验同时是参考了PV-RCNN文章中的waymo实验。但是非常的慢，达不到10FPS的要求，后续可以考虑apex加速等工作。但是two-stage的方法可能在这里不是很适用。可以看出在没有调参的基础上，在car和在truck上表现比较好。


 - 考虑实用性在FPS>10以上的测试
这部分实验主要通过修改官方给出的base_line，首先在满足FPS>10的基础上，做一定的参数调整和小的细节修改。

|model| 备注|FPS|MAP|
|--|--|--|--|
|second(30epoch)|det3D版本,<br>无gt_数据增广<br>设置[0.16, 0.16, 0.1],<br>max_points_in_voxel=10,|10.7| Evaluation official: car AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:68.53<br>3d   AP:58.96<br>aos  AP:0.00<br>car AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:70.86<br>3d   AP:70.74<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:46.35<br>3d   AP:35.10<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:60.20<br>3d   AP:51.22<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:37.10<br>3d   AP:27.48<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:49.20<br>3d   AP:48.59<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:51.28<br>3d   AP:50.27<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:60.13<br>3d   AP:60.08<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:25.82<br>3d   AP:23.94<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:39.67<br>3d   AP:39.41<br>aos  AP:0.00|
|second(30epoch)|det3D版本,<br><br>设置[0.16, 0.16, 0.1],<br>max_points_in_voxel=10,<br>增加gt数据增广<br>dict(Car=40),<br> dict(Truck=45),<br> dict(Tricar=45),<br>dict(Cyclist=45),<br>dict(Pedestrian=45)<br>|10.7|  Evaluation<br>official: car AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:57.69<br>3d   AP:46.94<br>aos  AP:0.00<br>car AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:69.17<br>3d   AP:68.61<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:31.42<br>3d   AP:19.37<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:52.18<br>3d   AP:44.65<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:22.14<br>3d   AP:10.91<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:38.36<br>3d   AP:33.31<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:45.78<br>3d   AP:43.43<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:57.38<br><br>3d   AP:56.97<br><br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:28.24<br>3d   AP:23.63<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:40.77<br>3d   AP:40.33<br>aos  AP:0.00 |<br>
|second(30epoch)|det3D版本,<br>     voxel_size=[0.12, 0.12, 0.2],,<br>修改backbone使得z轴下采样为8倍<br>max_points_in_voxel=7,<br>增加gt数据增广<br>dict(Car=40),<br> dict(Truck=45),<br> dict(Tricar=45),<br>dict(Cyclist=45),<br>dict(Pedestrian=45)|10.7|  Evaluation official: car AP(Average Precision)@0.70, 0.70, 0.70<br>bbox AP:0.00<br>bev  AP:56.97<br>3d   AP:45.90<br>aos  AP:0.00<br>car AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:68.97<br>3d   AP:68.24<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>3d   AP:17.61<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:50.59<br>3d   AP:42.87<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:19.91<br>3d   AP:9.14<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:37.78<br>3d   AP:32.89<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:51.05<br>3d   AP:43.75<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:64.20<br>3d   AP:63.36<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:32.24<br>3d   AP:25.50<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:44.06<br>3d   AP:43.64<br>aos  AP:0.00 |
|second(30epoch)|det3D版本,<br>     voxel_size=[0.30, 0.30, 0.2],,<br>max_points_in_voxel=30,<br>增加gt数据增广<br>dict(Car=20),<br> dict(Truck=15),<br> dict(Tricar=15),<br>dict(Cyclist=15),<br>dict(Pedestrian=15)<br>增加VFE模块（16*32）|12|  Evaluation official: car AP(Average Precision)70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:47.00<br>3d   AP:36.51<br>aos  AP:0.00<br>car AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:66.31<br>3d   AP:58.74<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:24.46<br>3d   AP:12.94<br>aos  AP:0.00<br>truck AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:46.98<br>3d   AP:43.84<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.70, 0.70:<br>bbox AP:0.00<br>bev  AP:16.55<br>3d   AP:12.30<br>aos  AP:0.00<br>tricar AP(Average Precision)@0.70, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:29.58<br>3d   AP:24.20<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:42.81<br>3d   AP:35.55<br>aos  AP:0.00<br>cyclist AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:54.78<br>3d   AP:54.31<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.50, 0.50:<br>bbox AP:0.00<br>bev  AP:13.37<br>3d   AP:12.05<br>aos  AP:0.00<br>pedestrian AP(Average Precision)@0.50, 0.25, 0.25:<br>bbox AP:0.00<br>bev  AP:19.18<br>3d   AP:18.84<br>aos  AP:0.00 |

 - FPS>10在网络上的改进

|model| 备注|FPS|MAP|
|--|--|--|--|
|second+分割+中心预测(30epoch)|det3D版本,<br>无gt_数据增广<br>设置[0.16, 0.16, 0.1],<br>max_points_in_voxel=10,<br>加SA-SSD的分割和中心预测|10.7| 

# 在华为云上部署
需要的文件（环境、数据、infos、db_datasets） [下载](https://drive.google.com/drive/folders/142xG50Dg3BivDodzswKGZi2zlIfU9yOh?usp=sharing)
## Dataset Preparation
在华为云上的文件路径安排如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705064634477.png)
其中baseline如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200705064723135.png)

DeepCamp_Lidar数据准备如下：

```
── DeepCamp_Lidar
   │──labels_filter
   │──test_filter
   │──test_video_filter
   │── train_val_filter
   │── dee_infos_val.pkl
   │── dee_infos_trainval.pkl
   │── dee_infos_train.pkl
   │── dee_dbinfos_train.pkl
   │── dee_dbinfos_test.pkl
   │── dee_dbinfos_test.pkl
   │── dee_infos_test_video1.pkl
   │── dee_infos_test_video2.pkl
   │── gt_database
```

pvrcnn


```
pvrcnn
├── OpenLidarPerceptron
```
## 训练
### pv-rcnn
预训练模型，针对car训练了一个模型 [car_ckpt](https://drive.google.com/file/d/1hDdZPWY6IbTdnroKg2z7sI6XsvnyZP2T/view?usp=sharing)，针对所有物体训练了一个模型 [all_ckpt](https://drive.google.com/file/d/1d_UcladBOv4VPUY2PBUM11ZP2lM7Dj1s/view?usp=sharing) .

单卡 训练模型：

```
python train.py --cfg_file ./cfgs/second_dee_car.yaml --batch_size 2 
```
多卡训练：

```
python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file ./cfgs/second_deecamp.yaml --batch_size 4
```
### det3D
单卡训练

```
python3.6 tools/train.py ./examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir ./res
```
多卡

```
python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch ../examples/second/configs/deepcamp_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py  --work_dir ./ress
```

## 测试
### pvrcnn
```
 python test.py --cfg_file ../cfgs/second_deecamp.yaml --batch_size 4 --ckpt /your/path//checkpoint_epoch_20.pth
```

# 在个人服务器上部署

 - [ ] 0.0.
