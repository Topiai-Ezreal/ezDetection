from mmdet.apis import init_detector, inference_detector, show_result_pyplot


config_file = 'configs\\faster_rcnn\\faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints\\epoch_20.pth'
#checkpoint_file = 'checkpoints\\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cpu'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = 'demo\\000068.jpeg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result)



'''
img = cv2.imread('demo/demo.jpg')
cv2.imshow('show', img)
cv2.waitKey(0)
'''
