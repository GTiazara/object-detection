# import cv2
# from PIL import Image

# from ultralytics import YOLO

# model = YOLO("//home/GTiazara/Téléchargements/best.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# # results = model.predict(source="0")
# # results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# # from PIL
# im1 = Image.open("/home/GTiazara/Documents/workspace/get_experience_project/airplane-detection/data/input/0_1.tif")
# results = model.predict(source=im1, save=True)  # save plotted images

# # from ndarray
# im2 = cv2.imread("/home/GTiazara/Documents/workspace/get_experience_project/airplane-detection/data/input/1_1.tif")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])

from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = 'test.jpg'  

result = inference_detector(model, img)

model.show_result(img, result, out_file='result.jpg')