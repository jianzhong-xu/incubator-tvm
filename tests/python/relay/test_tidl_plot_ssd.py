import os
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import numpy as np

img           = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './tidl_outputs/street_small.npy')).astype(np.float32)
class_IDs     = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './tidl_outputs/out_tidl_0.npy')).astype(np.float32)
scores        = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './tidl_outputs/out_tidl_1.npy')).astype(np.float32)
bounding_boxs = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './tidl_outputs/out_tidl_2.npy')).astype(np.float32)

#model_name = "ssd_512_mobilenet1.0_coco"
model_name = "ssd_512_mobilenet1.0_voc"
block = model_zoo.get_model(model_name, pretrained=True)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=block.classes)
plt.savefig("gluoncv_ssd_tidl_0.png")

#ax = utils.viz.plot_bbox(img, bounding_boxs[1], scores[1],
#                         class_IDs[1], class_names=block.classes)
#plt.savefig("gluoncv_ssd_tidl_1.png")
