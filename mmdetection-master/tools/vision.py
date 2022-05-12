import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent
import os


async def main():
    config_file = 'E:\jlf\c\\config.py'
    checkpoint_file = 'E:\jlf\c\\epoch_8.pth'
    device = 'cpu'
    # model = init_detector(config_file, checkpoint=checkpoint_file, device=device)
    model = torch.load(checkpoint_file, map_location='cpu')
    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    # for _ in range(streamqueue_size):
    #     streamqueue.put_nowait(torch.cuda.Stream())

    img_path = 'E:\jlf\c\\raw'
    save_path = 'E:\jlf\c\\1100'
    for name in os.listdir(img_path):
        img = os.path.join(img_path, name)
        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)
        model.show_result(img, result, out_file=os.path.join(save_path, name))


asyncio.run(main())

# /home/jiaolinfei/.conda/envs/mmDetection/bin/python /home/jiaolinfei/mmdetection-master/work_dirs/learn_code/vision_demo.py
# python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --show
