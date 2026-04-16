"""Script for running inference on a single image using YOLOv7."""
# pylint: disable=import-error,no-name-in-module
from pathlib import Path
from time import perf_counter

from importlib_resources import files

import cv2
import torch

from yolov7.yolov7 import YOLOv7


imgpath = Path('/workspace/test.jpeg')
if not imgpath.is_file():
    raise AssertionError(f'{str(imgpath)} not found')

OUTPUT_FOLDER = 'inference'
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

yolov7 = YOLOv7(
    weights=files('yolov7').joinpath('weights/yolov7-w6_last_state.pt'),
    cfg=files('yolov7').joinpath('cfg/deploy/yolov7-w6.yaml'),
    bgr=True,
    device='cuda',
    model_image_size=640,
    max_batch_size=64,
    half=True,
    same_size=True,
    conf_thresh=0.25,
    trace=False,
    cudnn_benchmark=False,
)

img = cv2.imread(str(imgpath))
BATCH_SIZE = 512
imgs = [img for _ in range(BATCH_SIZE)]

NUM_ITERATIONS = 3
DUR = 0
for i in range(NUM_ITERATIONS):
    torch.cuda.synchronize()
    tic = perf_counter()
    dets = yolov7.detect_get_box_in(imgs, box_format='ltrb', classes=None, buffer_ratio=0.0)[0]
    # dets = yolov7.detect_get_box_in(imgs, box_format='ltrb', classes=['person'], buffer_ratio=0.0)[0]
    # print('detections: {}'.format(dets))
    torch.cuda.synchronize()
    toc = perf_counter()
    if i > 1:
        DUR += toc - tic
print(f'Average time taken: {(DUR/NUM_ITERATIONS*1000):0.2f}ms')

draw_frame = img.copy()
for det in dets:
    # print(det)
    bb, score, class_ = det
    l, t, r, b = bb
    cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
    cv2.putText(draw_frame, class_, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

output_path = Path(OUTPUT_FOLDER) / 'test_out.jpg'
cv2.imwrite(str(output_path), draw_frame)
