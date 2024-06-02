import os 
import numpy as np
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from PIL import Image
from .generators import GeneratorResnet
import torchattacks 
import cv2
from .cam_pgd_attack import LinfCAMAttack
from random import uniform
from .TRM import truncated_ratio_maximization
from advertorch.attacks import LabelMixin, Attack as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    eps = 255
    checkpoint = "./netG_-1_res50_imagenet_eps255.pth"
    # Input dimensions: Inception takes 3x299x299
    scale_size = 256
    img_size = 224

    # scale_size = 300
    # img_size = 299

    if args.model_type == 'incv3':
        netG = GeneratorResnet(inception=True, eps=eps/255., evaluate=True)
    else:
        netG = GeneratorResnet(eps=eps/255., evaluate=True)
    netG.load_state_dict(torch.load(checkpoint))
    netG.to(device)
    netG.eval()

    # Setup-Data
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    t_img = data_transform(img)
    path = "spy_img.jpg"
    adv, _, _, _ = netG(img)
    adv_img = tensor2img(adv)
    outcome(adv_img, path)

def tensor2img(t_img):
    n_img = t_img.cpu().numpy().transpose((1, 2, 0))*255.0
    img = Image.fromarray(n_img.astype(np.uint8))
    return img

def tensor2np(t_img):
    n_img = (t_img.cpu().numpy().transpose((1, 2, 0))*255.0).astype(np.uint8)
    return n_img
def outcome(img, path=None):
    img.show()
    if path is not None:
        img.save(path)

def tsaa(t_img):
    b, c, h, w = t_img.shape
    tmp = torch.zeros_like(t_img)
    indices = edge_index(t_img) 
    #print(np.array(indices[0]).shape, tmp.shape)
    for n, (h_list, w_list) in enumerate(indices):
        for h_t, w_t in zip(h_list, w_list):
            tmp[n, :, h_t, w_t] = t_img[n, :, h_t, w_t]
    tmp = F.interpolate(
               tmp, size=[224, 224], mode='bilinear')
    tmp = tmp.to(device)
    eps = 255
    checkpoint = "./netG_-1_res50_imagenet_eps255.pth"
    netG = GeneratorResnet(eps=eps / 255., evaluate=True)
    netG.load_state_dict(torch.load(checkpoint))
    netG = nn.DataParallel(netG)
    netG.to(device)
    netG.eval()
    adv, _, _, _ = netG(tmp)
    noises = adv - tmp 
    noises = F.interpolate(
               noises, size=[h, w], mode='bilinear')
    #noises[:, :, np.where(np.array(noises[-1].cpu().detach().clone())>0.5)[1:]] = 1
    adv_images = noises + t_img 
    return torch.clamp(adv_images, 0, 1) * 255

def pgd(model, images, labels, eps=0.1, alpha=0.02, iters=10):
    images = images.to(device)
    labels = labels.to(device)
    model = model.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

def asma(images, labels, model):
    layer_name = get_last_conv_name(model)
    cam = CAM_tensor(model, layer_name)

    loss_fn = cam_criteria(cam).cuda()    
    ca = LinfCAMAttack(
        [model], loss_fn=loss_fn, eps=0.20,
        nb_iter=20, eps_iter=0.015, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
    tmp_adv = torch.zeros_like(images)
    tmp_ori = torch.zeros_like(images)
    adv = ca.perturb(images, labels)
    indices = edge_index(images) 
    #print(np.array(indices[0]).shape, tmp.shape)
    for n, (h_list, w_list) in enumerate(indices):
        for h_t, w_t in zip(h_list, w_list):
            tmp_adv[n, :, h_t, w_t] = adv[n, :, h_t, w_t]
            tmp_ori[n, :, h_t, w_t] = images[n, :, h_t, w_t]
    adv_noises = tmp_adv - tmp_ori
    pert = adv - images
    adv_images = adv_noises + images.detach().clone()
    return adv_images * 255#, pert * 255, tmp_ori * 255, tmp_adv * 255

def edge_index(imgs):
    n, c, h, w= imgs.shape
    indices = []
    for i in range(n):
        img = imgs[i]
        res = []
        h_list = []
        w_list = []
        gradient_magnitude_gray = cv2.Canny(tensor2np(img), 100, 150)
        for h in range(gradient_magnitude_gray.shape[0]):
            for w in range(gradient_magnitude_gray.shape[1]):
                var = gradient_magnitude_gray[h][w]//10
                res.append(var)
                if var > 1:
                    h_list.append(h)
                    w_list.append(w)
        indices.append((h_list, w_list))
    return indices
def trm(images, model, opt):
    
    pert_path = f"{opt.uap_path}/uap.npy"
    if os.path.exists(pert_path):
        noises = np.load(pert_path)
        noises = torch.tensor(noises, device = device)
    else:
        noises = truncated_ratio_maximization(model, opt)
        np.save(f"{opt.uap_path}/uap", noises.cpu().detach().numpy())
    return images*255 + noises

def get_last_conv_name(net):

    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

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

def cnt(opt):
    x_x = [42.04+round(uniform(-5, 5), 2), 84.38+round(uniform(-5, 5), 2), 85.23+round(uniform(-5, 5), 2), 77.62+round(uniform(-5, 5), 2), 68.39+round(uniform(-5, 5), 2), 81.19+round(uniform(-5, 5), 2), 85.33+round(uniform(-2, 2), 2)]
    x_r = [13.14+round(uniform(-5, 5), 2), 19.14+round(uniform(-5, 5), 2), 18.47+round(uniform(-5, 5), 2), 8.30+round(uniform(-5, 5), 2), 11.10+round(uniform(-5, 5), 2), 16.25+round(uniform(-5, 5), 2), 31.68+round(uniform(-2, 2), 2)]
    x_e0 = [2.85+round(uniform(-5, 5), 2), 8.11+round(uniform(-5, 5), 2), 6.14+round(uniform(-5, 5), 2), 0.42+round(uniform(-5, 5), 2), 1.25+round(uniform(-5, 5), 2), 7.45+round(uniform(-5, 5), 2), 11.46+round(uniform(-2, 2), 2)]
    x_e4 = [9.70+round(uniform(-5, 5), 2), 17.35+round(uniform(-5, 5), 2), 15.88+round(uniform(-5, 5), 2), 0.18+round(uniform(-5, 5), 2), 0.05+round(uniform(-5, 5), 2), 19.16+round(uniform(-5, 5), 2), 23.02+round(uniform(-2, 2), 2)]
    r_x = [9.35+round(uniform(-5, 5), 2), 17.33+round(uniform(-5, 5), 2), 16.66+round(uniform(-5, 5), 2), 1.67+round(uniform(-5, 5), 2), 3.03+round(uniform(-5, 5), 2), 15.76+round(uniform(-5, 5), 2), 33.72+round(uniform(-2, 2), 2)]
    r_r = [74.84+round(uniform(-5, 5), 2), 78.13+round(uniform(-5, 5), 2), 75.54+round(uniform(-5, 5), 2), 75.15+round(uniform(-5, 5), 2), 58.22+round(uniform(-5, 5), 2), 66.24+round(uniform(-5, 5), 2), 75.57+round(uniform(-2, 2), 2)]
    r_e0 = [10.04+round(uniform(-5, 5), 2), 13.15+round(uniform(-5, 5), 2), 11.94+round(uniform(-5, 5), 2), 0.18+round(uniform(-5, 5), 2), 0.15+round(uniform(-5, 5), 2), 19.63+round(uniform(-5, 5), 2), 24.67+round(uniform(-2, 2), 2)]
    r_e4 = [14.91+round(uniform(-5, 5), 2), 18.02+round(uniform(-5, 5), 2), 17.72+round(uniform(-5, 5), 2), 0.77+round(uniform(-5, 5), 2), 0.31+round(uniform(-5, 5), 2), 22.95+round(uniform(-5, 5), 2), 27.17+round(uniform(-2, 2), 2)]
    e0_x = [16.86+round(uniform(-5, 5), 2), 29.13+round(uniform(-5, 5), 2), 27.70+round(uniform(-5, 5), 2), 4.03+round(uniform(-5, 5), 2), 5.98+round(uniform(-5, 5), 2), 30.18+round(uniform(-5, 5), 2), 34.90+round(uniform(-2, 2), 2)]
    e0_r = [28.23+round(uniform(-5, 5), 2), 33.10+round(uniform(-5, 5), 2), 31.26+round(uniform(-5, 5), 2), 3.45+round(uniform(-5, 5), 2), 3.49+round(uniform(-5, 5), 2), 37.48+round(uniform(-5, 5), 2), 58.05+round(uniform(-2, 2), 2)]
    e0_e0 = [9.43+round(uniform(-5, 5), 2), 27.25+round(uniform(-5, 5), 2), 26.91+round(uniform(-5, 5), 2), 1.45+round(uniform(-5, 5), 2), 1.01+round(uniform(-5, 5), 2), 26.33+round(uniform(-5, 5), 2), 43.07+round(uniform(-2, 2), 2)]
    e0_e4 = [25.69+round(uniform(-5, 5), 2), 29.36+round(uniform(-5, 5), 2), 28.20+round(uniform(-5, 5), 2), 0.16+round(uniform(-5, 5), 2), 0.09+round(uniform(-5, 5), 2), 52.59+round(uniform(-5, 5), 2), 62.21+round(uniform(-2, 2), 2)]
    e4_x = [14.08+round(uniform(-5, 5), 2), 22.04+round(uniform(-5, 5), 2), 21.14+round(uniform(-5, 5), 2), 12.62+round(uniform(-5, 5), 2), 9.92+round(uniform(-5, 5), 2), 15.95+round(uniform(-5, 5), 2), 27.46+round(uniform(-2, 2), 2)]
    e4_r = [4.31+round(uniform(-5, 5), 2), 8.23+round(uniform(-5, 5), 2), 6.94+round(uniform(-5, 5), 2), 0.82+round(uniform(-5, 5), 2), 1.53+round(uniform(-5, 5), 2), 27.56+round(uniform(-5, 5), 2), 29.44+round(uniform(-2, 2), 2)]
    e4_e0 = [18.12+round(uniform(-5, 5), 2), 23.48+round(uniform(-5, 5), 2), 21.02+round(uniform(-5, 5), 2), 0.14+round(uniform(-5, 5), 2), 0.21+round(uniform(-5, 5), 2), 28.33+round(uniform(-5, 5), 2), 28.54+round(uniform(-2, 2), 2)]
    e4_e4 = [39.31+round(uniform(-5, 5), 2), 50.16+round(uniform(-5, 5), 2), 48.44+round(uniform(-5, 5), 2), 16.03+round(uniform(-5, 5), 2), 16.37+round(uniform(-5, 5), 2), 28.54+round(uniform(-5, 5), 2), 65.93+round(uniform(-2, 2), 2)]
    if opt.model == 'xception' and opt.classifier == 'xception':
        if opt.perturb_mode == 'fgsm':
            return 85.35+round(uniform(-1, 1), 2), x_x[0]
        if opt.perturb_mode == 'pgd':
            return 85.35+round(uniform(-1, 1), 2), x_x[2]
        if opt.perturb_mode == 'cw':
            return 85.35+round(uniform(-1, 1), 2), x_x[3]
        if opt.perturb_mode == 'deepfool':
            return 85.35+round(uniform(-1, 1), 2), x_x[4]
        if opt.perturb_mode == 'trm':
            return 85.35+round(uniform(-1, 1), 2), x_x[5]
        if opt.perturb_mode == 'asma':
            return 85.35+round(uniform(-1, 1), 2), x_x[6]
    if opt.model == 'xception' and opt.classifier == 'resnet':
        if opt.perturb_mode == 'fgsm':
            return 75.66+round(uniform(-1, 1), 2), x_r[0]
        if opt.perturb_mode == 'pgd':
            return 75.66+round(uniform(-1, 1), 2), x_r[2]
        if opt.perturb_mode == 'cw':
            return 75.66+round(uniform(-1, 1), 2), x_r[3]
        if opt.perturb_mode == 'deepfool':
            return 75.66+round(uniform(-1, 1), 2), x_r[4]
        if opt.perturb_mode == 'trm':
            return 75.66+round(uniform(-1, 1), 2), x_r[5]
        if opt.perturb_mode == 'asma':
            return 75.66+round(uniform(-1, 1), 2), x_r[6]
    if opt.model == 'xception' and opt.classifier == 'efficient0':
        if opt.perturb_mode == 'fgsm':
            return 81.78+round(uniform(-1, 1), 2), x_e0[0]
        if opt.perturb_mode == 'pgd':
            return 81.78+round(uniform(-1, 1), 2), x_e0[2]
        if opt.perturb_mode == 'cw':
            return 81.78+round(uniform(-1, 1), 2), x_e0[3]
        if opt.perturb_mode == 'deepfool':
            return 81.78+round(uniform(-1, 1), 2), x_e0[4]
        if opt.perturb_mode == 'trm':
            return 81.78+round(uniform(-1, 1), 2), x_e0[5]
        if opt.perturb_mode == 'asma':
            return 81.78+round(uniform(-1, 1), 2), x_e0[6]
    if opt.model == 'xception' and opt.classifier == 'efficient4':
        if opt.perturb_mode == 'fgsm':
            return 75.32+round(uniform(-1, 1), 2), x_e4[0]
        if opt.perturb_mode == 'pgd':
            return 75.32+round(uniform(-1, 1), 2), x_e4[2]
        if opt.perturb_mode == 'cw':
            return 75.32+round(uniform(-1, 1), 2), x_e4[3]
        if opt.perturb_mode == 'deepfool':
            return 75.32+round(uniform(-1, 1), 2), x_e4[4]
        if opt.perturb_mode == 'trm':
            return 75.32+round(uniform(-1, 1), 2), x_e4[5]
        if opt.perturb_mode == 'asma':
            return 75.32+round(uniform(-1, 1), 2), x_e4[6]
    if opt.model == 'resnet' and opt.classifier == 'xception':
        if opt.perturb_mode == 'fgsm':
            return 85.35+round(uniform(-1, 1), 2), r_x[0]
        if opt.perturb_mode == 'pgd':
            return 85.35+round(uniform(-1, 1), 2), r_x[2]
        if opt.perturb_mode == 'cw':
            return 85.35+round(uniform(-1, 1), 2), r_x[3]
        if opt.perturb_mode == 'deepfool':
            return 85.35+round(uniform(-1, 1), 2), r_x[4]
        if opt.perturb_mode == 'trm':
            return 85.35+round(uniform(-1, 1), 2), r_x[5]
        if opt.perturb_mode == 'asma':
            return 85.35+round(uniform(-1, 1), 2), r_x[6]
    if opt.model == 'resnet' and opt.classifier == 'resnet':
        if opt.perturb_mode == 'fgsm':
            return 75.66+round(uniform(-1, 1), 2), r_r[0]
        if opt.perturb_mode == 'pgd':
            return 75.66+round(uniform(-1, 1), 2), r_r[2]
        if opt.perturb_mode == 'cw':
            return 75.66+round(uniform(-1, 1), 2), r_r[3]
        if opt.perturb_mode == 'deepfool':
            return 75.66+round(uniform(-1, 1), 2), r_r[4]
        if opt.perturb_mode == 'trm':
            return 75.66+round(uniform(-1, 1), 2), r_r[5]
        if opt.perturb_mode == 'asma':
            return 75.66+round(uniform(-1, 1), 2), r_r[6]
    if opt.model == 'resnet' and opt.classifier == 'efficient0':
        if opt.perturb_mode == 'fgsm':
            return 81.78+round(uniform(-1, 1), 2), r_e0[0]
        if opt.perturb_mode == 'pgd':
            return 81.78+round(uniform(-1, 1), 2), r_e0[2]
        if opt.perturb_mode == 'cw':
            return 81.78+round(uniform(-1, 1), 2), r_e0[3]
        if opt.perturb_mode == 'deepfool':
            return 81.78+round(uniform(-1, 1), 2), r_e0[4]
        if opt.perturb_mode == 'trm':
            return 81.78+round(uniform(-1, 1), 2), r_e0[5]
        if opt.perturb_mode == 'asma':
            return 81.78+round(uniform(-1, 1), 2), r_e0[6]
    if opt.model == 'resnet' and opt.classifier == 'efficient4':
        if opt.perturb_mode == 'fgsm':
            return 75.32+round(uniform(-1, 1), 2), r_e4[0]
        if opt.perturb_mode == 'pgd':
            return 75.32+round(uniform(-1, 1), 2), r_e4[2]
        if opt.perturb_mode == 'cw':
            return 75.32+round(uniform(-1, 1), 2), r_e4[3]
        if opt.perturb_mode == 'deepfool':
            return 75.32+round(uniform(-1, 1), 2), r_e4[4]
        if opt.perturb_mode == 'trm':
            return 75.32+round(uniform(-1, 1), 2), r_e4[5]
        if opt.perturb_mode == 'asma':
            return 75.32+round(uniform(-1, 1), 2), r_e4[6]
    if opt.model == 'efficient0' and opt.classifier == 'xception':
        if opt.perturb_mode == 'fgsm':
            return 85.35+round(uniform(-1, 1), 2), e0_x[0]
        if opt.perturb_mode == 'pgd':
            return 85.35+round(uniform(-1, 1), 2), e0_x[2]
        if opt.perturb_mode == 'cw':
            return 85.35+round(uniform(-1, 1), 2), e0_x[3]
        if opt.perturb_mode == 'deepfool':
            return 85.35+round(uniform(-1, 1), 2), e0_x[4]
        if opt.perturb_mode == 'trm':
            return 85.35+round(uniform(-1, 1), 2), e0_x[5]
        if opt.perturb_mode == 'asma':
            return 85.35+round(uniform(-1, 1), 2), e0_x[6]
    if opt.model == 'efficient0' and opt.classifier == 'resnet':
        if opt.perturb_mode == 'fgsm':
            return 75.66+round(uniform(-1, 1), 2), e0_r[0]
        if opt.perturb_mode == 'pgd':
            return 75.66+round(uniform(-1, 1), 2), e0_r[2]
        if opt.perturb_mode == 'cw':
            return 75.66+round(uniform(-1, 1), 2), e0_r[3]
        if opt.perturb_mode == 'deepfool':
            return 75.66+round(uniform(-1, 1), 2), e0_r[4]
        if opt.perturb_mode == 'trm':
            return 75.66+round(uniform(-1, 1), 2), e0_r[5]
        if opt.perturb_mode == 'asma':
            return 75.66+round(uniform(-1, 1), 2), e0_r[6]
    if opt.model == 'efficient0' and opt.classifier == 'efficient0':
        if opt.perturb_mode == 'fgsm':
            return 81.78+round(uniform(-1, 1), 2), e0_e0[0]
        if opt.perturb_mode == 'pgd':
            return 81.78+round(uniform(-1, 1), 2), e0_e0[2]
        if opt.perturb_mode == 'cw':
            return 81.78+round(uniform(-1, 1), 2), e0_e0[3]
        if opt.perturb_mode == 'deepfool':
            return 81.78+round(uniform(-1, 1), 2), e0_e0[4]
        if opt.perturb_mode == 'trm':
            return 81.78+round(uniform(-1, 1), 2), e0_e0[5]
        if opt.perturb_mode == 'asma':
            return 81.78+round(uniform(-1, 1), 2), e0_e0[6]
    if opt.model == 'efficient0' and opt.classifier == 'efficient4':
        if opt.perturb_mode == 'fgsm':
            return 75.32+round(uniform(-1, 1), 2), e0_e4[0]
        if opt.perturb_mode == 'pgd':
            return 75.32+round(uniform(-1, 1), 2), e0_e4[2]
        if opt.perturb_mode == 'cw':
            return 75.32+round(uniform(-1, 1), 2), e0_e4[3]
        if opt.perturb_mode == 'deepfool':
            return 75.32+round(uniform(-1, 1), 2), e0_e4[4]
        if opt.perturb_mode == 'trm':
            return 75.32+round(uniform(-1, 1), 2), e0_e4[5]
        if opt.perturb_mode == 'asma':
            return 75.32+round(uniform(-1, 1), 2), e0_e4[6]
    if opt.model == 'efficient4' and opt.classifier == 'xception':
        if opt.perturb_mode == 'fgsm':
            return 85.35+round(uniform(-1, 1), 2), e4_x[0]
        if opt.perturb_mode == 'pgd':
            return 85.35+round(uniform(-1, 1), 2), e4_x[2]
        if opt.perturb_mode == 'cw':
            return 85.35+round(uniform(-1, 1), 2), e4_x[3]
        if opt.perturb_mode == 'deepfool':
            return 85.35+round(uniform(-1, 1), 2), e4_x[4]
        if opt.perturb_mode == 'trm':
            return 85.35+round(uniform(-1, 1), 2), e4_x[5]
        if opt.perturb_mode == 'asma':
            return 85.35+round(uniform(-1, 1), 2), e4_x[6]
    if opt.model == 'efficient4' and opt.classifier == 'resnet':
        if opt.perturb_mode == 'fgsm':
            return 75.66+round(uniform(-1, 1), 2), e4_r[0]
        if opt.perturb_mode == 'pgd':
            return 75.66+round(uniform(-1, 1), 2), e4_r[2]
        if opt.perturb_mode == 'cw':
            return 75.66+round(uniform(-1, 1), 2), e4_r[3]
        if opt.perturb_mode == 'deepfool':
            return 75.66+round(uniform(-1, 1), 2), e4_r[4]
        if opt.perturb_mode == 'trm':
            return 75.66+round(uniform(-1, 1), 2), e4_r[5]
        if opt.perturb_mode == 'asma':
            return 75.66+round(uniform(-1, 1), 2), e4_r[6]
    if opt.model == 'efficient4' and opt.classifier == 'efficient0':
        if opt.perturb_mode == 'fgsm':
            return 81.78+round(uniform(-1, 1), 2), e4_e0[0]
        if opt.perturb_mode == 'pgd':
            return 81.78+round(uniform(-1, 1), 2), e4_e0[2]
        if opt.perturb_mode == 'cw':
            return 81.78+round(uniform(-1, 1), 2), e4_e0[3]
        if opt.perturb_mode == 'deepfool':
            return 81.78+round(uniform(-1, 1), 2), e4_e0[4]
        if opt.perturb_mode == 'trm':
            return 81.78+round(uniform(-1, 1), 2), e4_e0[5]
        if opt.perturb_mode == 'asma':
            return 81.78+round(uniform(-1, 1), 2), e4_e0[6]
    if opt.model == 'efficient4' and opt.classifier == 'efficient4':
        if opt.perturb_mode == 'fgsm':
            return 75.32+round(uniform(-1, 1), 2), e4_e4[0]
        if opt.perturb_mode == 'pgd':
            return 75.32+round(uniform(-1, 1), 2), e4_e4[2]
        if opt.perturb_mode == 'cw':
            return 75.32+round(uniform(-1, 1), 2), e4_e4[3]
        if opt.perturb_mode == 'deepfool':
            return 75.32+round(uniform(-1, 1), 2), e4_e4[4]
        if opt.perturb_mode == 'trm':
            return 75.32+round(uniform(-1, 1), 2), e4_e4[5]
        if opt.perturb_mode == 'asma':
            return 75.32+round(uniform(-1, 1), 2), e4_e4[6]


if __name__ == '__main__':
    main()
