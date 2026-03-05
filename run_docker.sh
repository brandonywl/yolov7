WORKSPACE=/media/data/yolov7
DATA=/media/data/datasets

# DATASET_LOCATION=/path/to/datasets_folder
DATASET_LOCATION=~/Downloads/coco128
DATASET_DESTINATION=/media/data/yolov7/coco128

# DATA2=/media/data/fdet-api

# docker run -it --rm \
# 	--gpus all \
#     -w $WORKSPACE \
# 	-v $WORKSPACE:$WORKSPACE \
# 	-v $DATA:$DATA \
# 	--shm-size=64g \
# 	--net host \
# 	yolov7_main

# -v $DATA2:$DATA2 \
# reid_pipeline_sahi_fr

docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ./:/yolov7 \
    -v $DATASET_LOCATION:$DATASET_DESTINATION \
    yolov7-main:latest bash