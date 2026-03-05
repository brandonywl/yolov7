# docker build -t yolov7-main:latest .

# # Target CUDA 12.8
FROM nvcr.io/nvidia/pytorch:25.03-py3

# CUDA 11.8.0
# FROM nvcr.io/nvidia/pytorch:22.12-py3

# CUDA 11.7.1
# FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/yolov7/"
WORKDIR $cwd

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX 8.9 9.0 10.0 12.0"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && apt-get -y upgrade

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

### APT END ###

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# RUN pip3 uninstall -y opencv-python opencv-python-headless
# RUN pip3 install opencv-python==4.8.0.74

# RUN pip3 install cython && \
#     pip3 install --no-build-isolation --no-cache-dir 'git+https://github.com/yhsmiley/fdet-api.git#subdirectory=PythonAPI'

# RUN apt-get remove -y python3-pycocotools || true
# RUN pip3 uninstall -y pycocotools
# RUN pip3 install --no-cache-dir --force-reinstall pycocotools==2.0.7
