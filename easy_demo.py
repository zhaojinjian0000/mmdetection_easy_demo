import torch,torchvision
print('torch',torch.__version__,torch.cuda.is_available())
import mmdet
print('mmdet',mmdet.__version__)
from mmcv.ops import get_compiling_cuda_version,get_compiler_version
print('get_compiling_cuda_version',get_compiling_cuda_version())
print('get_compiler_version',get_compiler_version())

from mmdet.apis import init_detector, inference_detector,show_result_pyplot
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file, checkpoint_file, device=device)
for name,module in model.named_children():
    print(name)
    for n, layer in module.named_children():
        print('\t',n)
for name,module in model.named_children():
    print('-'*70,name)
    for n, layer in module.named_children():
        print('-'*10,n)
        print(layer)

img_path = 'demo/demo.jpg'
result = inference_detector(model, img_path)
show_result_pyplot(model,img_path,result,score_thr=0.95,wait_time=0)
print(len(result)) # 类别数
print(result[0].shape) #目标数x5

from types import MethodType
def new_simple_test(self, img, img_metas, proposals=None, rescale=False):
    x = self.extract_feat(img)
    proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    if rescale:
        for proposals, meta in zip(proposal_list, img_metas):
            proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
    return [proposal.cpu().numpy() for proposal in proposal_list]
model.simple_test = MethodType(new_simple_test,model)
rpn_result = inference_detector(model, img_path)
from mmdet.core.visualization import imshow_det_bboxes
import mmcv
import numpy as np
img = mmcv.imread(img_path)
bboxes = np.vstack(rpn_result)
labels = np.zeros(bboxes.shape[0],dtype=int)
imshow_det_bboxes(img,bboxes,labels,None,
        class_names=['']*80,
        score_thr=0.9,
        bbox_color='green',
        thickness=0.5,
        font_size=3,
        show=True,
        wait_time=0.1)






