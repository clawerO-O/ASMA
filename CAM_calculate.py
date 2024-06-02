import io
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
from utils.cam_pgd_attack import LinfCAMAttack
from utils.BalancedDataParallel import BalancedDataParallel
from utils.parse_config import merge

net = models.resnet50(num_classes=2)
#net.fc = nn.Linear(in_features=net.fc.in_features, out_features=2)
#ckpt = torch.load('checkpoints/resnet_best.pth', map_location="cpu")
#net.load_state_dict(ckpt["state_dict"])
net.eval().cuda()
features_blobs = []
finalconv_name = 'layer4'
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_last_conv_name(net):

    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

class CAM_divide_tensor(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = {}
        self.weight = None
        self.handlers = []
        self._register_hook()
        self._get_weight()

    def _get_features_hook(self, module, input, output):
        self.feature[input[0].device] = output

    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):

        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        index = np.argmax(output.cpu().data.numpy(), axis=1)

        weight = torch.zeros((inputs.size(0), self.weight.size(1))).cuda()

        weight[:] = self.weight[index[:]]

        feature = []

        for i in self.feature:
            feature.append(self.feature[i].to(torch.device("cuda:0")))

        feature = torch.cat(feature[:], 0)

        feature  = torch.nn.functional.interpolate(feature , (32, 32), mode='bilinear')

        return feature, weight
        

class CAM_tensor(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = {}
        self.weight = None
        self.handlers = []
        self._register_hook()
        self._get_weight()

    def _get_features_hook(self, module, input, output):
        self.feature[input[0].device] = output

    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        index = np.argmax(output.cpu().data.numpy(), axis=1)

        weight = torch.zeros((inputs.size(0), self.weight.size(1))).cuda()

        weight[:] = self.weight[index[:]]

        feature = []

        for i in self.feature:
            feature.append(self.feature[i].to(torch.device("cuda:0")))

        feature = torch.cat(feature[:], 0)

        cam = feature * weight.unsqueeze(-1).unsqueeze(-1)  # [B,C,H,W]

        cam = torch.max(cam, torch.zeros(cam.size()).cuda())  # ReLU

        cam = cam.clone() - torch.min(cam.clone())
        cam = cam.clone() / torch.max(cam.clone())

        cam = torch.nn.functional.interpolate(cam, (32, 32), mode='bilinear')

        return cam


class cam_criteria(nn.Module):
    def __init__(self, cam):
        super(cam_criteria, self).__init__()
        self.cam = cam
        self.mse = nn.MSELoss()

    def forward(self, adv, org):

        mask_adv = self.cam(adv)
        mask_ori = self.cam(org)

        out = self.mse(mask_adv, mask_ori)
        return out


class cam_divide_criteria(nn.Module):
    def __init__(self, cam):
        super(cam_feature_criteria, self).__init__()
        # you can try other losses
        self.cam = cam
        self.mse = nn.MSELoss()

    def forward(self, adv, org):

        mask_adv, weight_adv = self.cam(adv)
        mask_ori, weight_ori = self.cam(org)

        out1 = self.mse(mask_adv, mask_ori)
        out2 = self.mse(weight_adv, weight_ori) # torch.sum(torch.abs(weight_adv - weight_ori)) / weight_adv.size(0)
        return out1, out2

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((299,299)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open('true_3.png').convert('RGB')
img_tensor = preprocess(img_pil).unsqueeze(0).cuda()
#img_variable = Variable(img_tensor)

layer_name = get_last_conv_name(net)
cam = CAM_tensor(net, layer_name)
loss_fn = cam_criteria(cam).cuda()
adversary = LinfCAMAttack(
        [net], loss_fn=loss_fn, eps=0.1,
        nb_iter=10, eps_iter=3/255, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
img_variable  = merge(img_tensor, adversary)
logit = net(img_variable)
#print(logit.shape)

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
#print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('true_3.png')
img = cv2.resize(img, (512,512))
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(512, 512)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
pixel_sum = result.shape[0]*result.shape[1]
pixel_area = np.where(result>100)[0].shape[0]
print("CAM vaule is",pixel_area/pixel_sum)
cv2.imwrite('adv_t3_cam.jpg', result)


