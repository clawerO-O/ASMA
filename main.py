import os
import csv
import time
import shutil
import random
import argparse
os.environ['CUDA_VISBLE_DEVICES']='0, 1, 2, 3'
from PIL import Image
import cv2 as cv
import numpy as np
import torch
from utils.att import cnt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import conf as config
from merge_face import merge#, noise_mask
from models.xceptionnet import xception
from torchvision import transforms as t
from torchvision.models import resnet50
from models.efficientnet import EfficientNet
import attacks
from utils.att import trm
from utils.cam_pgd_attack import LinfCAMAttack
from utils.BalancedDataParallel import BalancedDataParallel
#from mydeepfool import DeepFool 
#from carlini import CarliniWagnerL2 as CW

def main(opt):
    start = time.time()
    torch.backends.cudnn.benchmark = True
    my_transform = t.Compose([    
    t.Resize(299),
    t.ToTensor(), 
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = DFDCDataset(config.data_list, "test", config.data_root,
                              transform=my_transform, stable=True) 
    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_dataset, **kwargs)
    print(len(test_loader), len(test_loader.dataset))
    attack_model,attack_model_name = get_model(opt.model)
    eval_model,eval_model_name = get_model(opt.classifier)
    show_model(attack_model_name,eval_model_name)
    acc_record0 = []	
    acc_record1 = []
    acc_record2 = []
    acc_record3 = []
    acc_record4 = []
    acc_record5 = []
    acc_record6 = []
    acc_record7 = []
    if opt.perturb_mode == 'fgsm':
        ad = attacks.FGSM(attack_model, eps=8/255)
        #fgsm.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #fgsm.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if opt.perturb_mode == 'bim':
        ad = attacks.BIM(attack_model, eps=8/255, alpha=2/255, steps=10)
        #ifgsm.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #ifgsm.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if opt.perturb_mode == 'pgd':
        ad = attacks.PGD(attack_model, eps=8/255, alpha=2/255, steps=10, random_start=False)
        #pgd.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #pgd.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if opt.perturb_mode == 'cw':
        ad = attacks.CW(attack_model, c=1, kappa=0, steps=100, lr=0.01)
        #cw.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #cw.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if opt.perturb_mode == 'df':
        ad = attacks.DeepFool(attack_model, steps=100, overshoot=0.02)
        #df.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #df.set_normalization_used(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if opt.perturb_mode == 'asma':
        layer_name = get_last_conv_name(eval_model)
        cam = CAM_tensor(eval_model, layer_name)

        loss_fn = cam_criteria(cam).cuda()
        ad = LinfCAMAttack(
            [attack_model], loss_fn=loss_fn, eps=0.20,
           nb_iter=20, eps_iter=0.015, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)

    for count, (imgs, labels) in enumerate(test_loader):
        epoch_start = time.time()
        flag = False
        if imgs.shape[0]<config.batch_size:
            break 
        #if count == 100:
        #    break
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if count == 1:
            flag = True
        acc_record0.append(cal_accuracy(eval_model, imgs, labels))
        if opt.perturb_mode == 'asma':
            adv_imgs = merge(imgs, ad)
        elif opt.perturb_mode == 'trm':
            adv_imgs = trm(imgs, eval_model, opt)
        else:
            adv_imgs = ad(imgs, labels)
        acc_record1.append(cal_accuracy(eval_model, adv_imgs, labels))
        print("epoch_{} cost time: {:.2f}".format(count,time.time()-epoch_start))
        torch.cuda.empty_cache()

    show_accuracy(acc_record0, acc_record1, opt)
    end = time.time()
    print("total time used is {:.2f}".format(end-start))
root_path = config.save_dir
model_path = ["best_xception.pth","best_resnet.pth","best_efficient-b0.pth","best_efficient-b4.pth"]
save_path = "results/"
inverse = t.Compose([
    t.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]),
    #t.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
    #t.ToPILImage(),
    ])

def get_model(model_name):
    num_classes = 2
    if model_name == "XceptionNet" or model_name == "xceptionnet" or model_name == '0' or model_name == "xception":
        model = xception(num_classes, pretrained=None).cuda()
        ckpt = torch.load(root_path+model_path[0], map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model_name = "XceptionNet"
    elif model_name == "ResNet50" or model_name == 'resnet50' or model_name == '1' or model_name == "resnet":
        model = resnet50(num_classes)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)
        ckpt = torch.load(root_path+model_path[1], map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model_name = "ResNet50"
    elif model_name == "EfficientNet-b0" or model_name == "EfficientNet-b0" or model_name == '2' or model_name == "efficient0":
        model = EfficientNet.from_name('efficientnet-b0')
        ckpt = torch.load(root_path+model_path[2], map_location="cpu")
        model.load_state_dict(ckpt)
        model._fc = nn.Linear(in_features=model._fc.in_features, out_features=2)
        model_name = "EfficientNet-b0"
    elif model_name == "EfficientNet-b4" or model_name == "EfficientNet-b4" or model_name == '3' or model_name == "efficient4":
        model = EfficientNet.from_name('efficientnet-b4')
        ckpt = torch.load(root_path+model_path[3], map_location="cpu")
        model.load_state_dict(ckpt)
        model._fc = nn.Linear(in_features=model._fc.in_features, out_features=2)
        model_name = "EfficientNet-b4"
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    return model,model_name

def show_model(attack_model,eval_model):
    print("="*50)
    print("="*17,"loading models","="*17)
    print("attack model is {}".format(attack_model))
    print("eval model is {}".format(eval_model))
    print("="*50)

def save_image(imgs,method = None):
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    #print(imgs.shape)
    for idx in range(imgs.shape[0]):
        img = imgs[idx,:,:,:].detach()
        img = img.squeeze(0)#inverse(img.squeeze(0))
        image = t.ToPILImage()(img)
        image.save("ca_mask_%03d.png"%idx)
        continue
        image = img_image(img)
        #image = img.resize((512,512), Image.ANTIALIAS)
        #print(image)
        if method is None:
            cv.imwrite(os.path.join(save_path,("ori_" + str(idx).rjust(2, '0') + "_"+".png")), image)
            #image.save(os.path.join(save_path,("ori_" + str(idx).rjust(2, '0') + "_"+".png")))
        elif method == 'ca':
            cv.imwrite(os.path.join(save_path, (method + "_" + str(idx).rjust(2, '0') + "_" + ".png")), image)
        else:
            cv.imwrite(os.path.join(save_path, (method.attack + "_" + str(idx).rjust(2, '0') + "_" + ".png")), image)
            #image.save(os.path.join(save_path, (method.attack + "_" + str(idx).rjust(2, '0') + "_" + ".png")))

def img_image(img):
    #print(img.shape)
    img = np.transpose(img.cpu().numpy(),(1,2,0))
    image = (img*255).round().astype(np.uint8)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def cal_accuracy(eval_model, adv_imgs, labels):
    with torch.no_grad():
        outputs = eval_model(adv_imgs)
        preds = torch.argmax(outputs.data, 1)
        iter_acc = torch.sum(preds == labels).item() / len(preds)
    return  iter_acc

def show_accuracy(ori_acc_record, adv_acc_record, opt):
    ori_acc = np.mean(ori_acc_record)*100.
    adv_acc = np.mean(adv_acc_record)*100.
    #ori_acc, adv_accs = cnt(opt)
    print("="*50)
    print("="*17,"accuracy","="*17)
    print("Origin : acc = %.4f"% (ori_acc))
    print("Test_%s: attack_acc = %.4f" % (opt.perturb_mode, adv_acc))
    print("="*50)

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
        #print(feature.shape, weight.shape)
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



class DFDCDataset(Dataset):
    def __init__(self, data_csv, required_set, data_root="",
                 ratio=(0.25, 0.05), stable=False, transform=None):
        video_info = []
        data_list = []

        with open(data_csv) as fin:
            reader = csv.DictReader(fin)

            for row in reader:
                if row["set_name"] == required_set:
                    label = int(row["is_fake"])
                    n_frame = int(row["n_frame"])
                    select_frame = round(n_frame * ratio[label])

                    for sample_idx in range(select_frame):
                        data_list.append((len(video_info), sample_idx))

                    video_info.append({
                        "name": row["name"],
                        "label": label,
                        "n_frame": n_frame,
                        "select_frame": select_frame,
                    })
            video_files = os.listdir(data_root)
            test_info = []
            test_list = []
            for video_file in video_files:
                video_file_name = video_file.split('.')[0]
                for x in video_info:
                    if x["name"] == video_file_name:
                        sel_frame = x["select_frame"]
                        for sa in range(sel_frame):
                            test_list.append((len(test_info), sa))
                        test_info.append(x)
                        break
            self.stable = stable
            self.data_root = data_root
            self.video_info = test_info
            self.data_list = test_list
            self.transform = transform

    def __getitem__(self, index):
        video_idx, sample_idx = self.data_list[index]
        info = self.video_info[video_idx]

        if self.stable:
            frame_idx = info["n_frame"] * sample_idx // info["select_frame"]
        else:
            frame_idx = random.randint(0, info["n_frame"] - 1)

        image_path = os.path.join(self.data_root, info["name"],
                                  "%03d.jpg" % frame_idx)
        try:
            img = Image.open(image_path).convert("RGB")
        except OSError:
            img = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
        if self.transform is not None:
            #result = self.transform(image=(np.array(img).astype(np.float32)))
            #img = result["image"]
            img = self.transform(img)


        return img, info["label"]

    def __len__(self):
        return len(self.data_list)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASMA")
    parser.add_argument('--val-dataset', type=str, default='../frames/', help='val dataset path')
    parser.add_argument('--prior', default='gauss', choices=['gauss', 'jigsaw', None],
                        help='the range prior of perturbations')
    parser.add_argument('--prior_batch', default=1, help='the batch size of prior')
    parser.add_argument('--std', default=10, help='initialize the standard deviation of gaussian noise')
    parser.add_argument('--lam', default=1, help='the parameter of negative activation')
    parser.add_argument('--epsilon', default=10 / 255, help='the infinite norm limitation of UAP')
    parser.add_argument('--delta_size', default=32, help='the size of delta')
    parser.add_argument('--uap_lr', default=0.1, help='the leraning rate of UAP')
    parser.add_argument('--gauss_t0', default=400,
                        help='the threshold to adjust the increasing rate of standard deviation(gauss)')
    parser.add_argument('--uap_path', default='perturbations/', help='the path of UAP')
    parser.add_argument('--p_active', default=True, type=bool, help='maximize the positive activation the conv layer')
    parser.add_argument('--p_rate', default=0.8, type=float, help='positive proportion of conv layer used')
    parser.add_argument('--n_active', default=True, type=bool, help='minimize the negative activation the conv layer')
    parser.add_argument('--n_rate', default=0.8, type=float,
                        help='negative proportion of conv layer used(deactivation)')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--gauss_gamma', default=10, help='the step size(gauss)')
    parser.add_argument('--model', type=str, default='xception', help='model for generating adversarial examples')
    parser.add_argument('--classifier', type=str, default='xception', help='model for evaluating attack effectiveness of adversarial examples')
    parser.add_argument('--perturb-mode', type=str, default='asma', help='algorithms used to perform attack')
    opt = parser.parse_args()
    main(opt)
