# WORKSPACE=/media/data/yolov7
# DATA=/media/data/datasets

# docker run -it --rm \
# 	--gpus all \
#     -w $WORKSPACE \
# 	-v $WORKSPACE:$WORKSPACE \
# 	-v $DATA:$DATA \
# 	yolov7

docker run -v ./:/workspace -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 yolov7-inference:latest bash