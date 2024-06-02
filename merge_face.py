import torch
#from torchvision import transforms
import torch.nn.functional as F
from models.faceparsenet.FaceParse import FaceParseNet50 as fp50
from PIL import Image
import numpy as np
import cv2 as cv

dict = {'background':0, 'skin':1, 'nose':2, 'eye_g':3, 'l_eye':4, 'r_eye':5,
        'l_brow':6, 'r_brow':7, 'l_ear':8, 'r_ear':9, 'mouth':10, 'u_lip':11,
        'l_lip':12, 'hair':13, 'hat':14, 'ear_r':15, 'neck_l':16, 'neck':17,
        'cloth':18}

model_path = 'checkpoints/38_G.pth'
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
mean, std = torch.tensor(mean).reshape(3, 1, 1).cuda(), torch.tensor(std).reshape(3, 1, 1).cuda()
parsing_model = fp50(num_classes=19,pretrained=False)
ckpt = torch.load(model_path, map_location="cpu")
parsing_model.load_state_dict(ckpt)
parsing_model.cuda()
parsing_model.eval()

def merge(imgs, attack, flag=False, features = ['nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow']):
    #print(imgs)
    index = []
    for c in features:
        index.append(dict[c]) 
    batch_size = imgs.shape[0]
    indexs = []
    merged_imgs = []
    for i in range(batch_size):
        img = imgs[i].detach().unsqueeze(0).cuda()#(1, 3, 299, 299)
        with torch.no_grad():
            img_ = F.interpolate(
                     img, (512, 512), mode='bilinear', align_corners=True)
            output = parsing_model(img_)
            output = output[0][-1]
            #print(output.shape)
            output = F.interpolate(
                     output, (imgs.shape[2], imgs.shape[3]), mode='bilinear', align_corners=True)
            parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)#(244,244)
            #print(parsing.shape)
            for idx in index:    #specific class 
                parsing[parsing == idx] = 1e6
            index = np.where(parsing==1e6)
            parsing[parsing != 1e6] = 0
            parsing[parsing == 1e6] = 255 #mark the targeted part 
            #parsing = torch.from_numpy(parsing).unsqueeze(0).numpy()#(1,244,244)
        merged_imgs.append(parsing)
        indexs.append(index)
    #merged_imgs = (merged_imgs - mean) / std #normalization to offset attack's effect
    adv_mask = attack.perturb(imgs*std+mean, None)
    if flag:
        return (adv_mask-mean)/std
# * torch.tensor(merged_imgs, dtype=torch.float32, device=torch.device('cuda:0')),0,1)
    #attack.perturb(torch.tensor(merged_imgs, dtype=torch.float32, device=torch.device('cuda:0')), None)#[16,1,244,244]
    #((imgs*std+mean), None)
    out_imgs = []
    #out_imgs = imgs*std+mean + adv_mask -mean
    #return out_imgs/std
    for i in range(batch_size):   
        img = imgs[i].detach()
        img = (img * std + mean)
        mask = adv_mask[i].detach()
        out_img = img.clone().detach()
        index = indexs[i]
        out_img[:, index[0],index[1]] = mask[:, index[0], index[1]]# + img[:, index[0], index[1]] 
        out_img = (out_img - mean) / std  
        out_imgs.append(out_img.cpu().numpy())
    out_imgs = torch.tensor(out_imgs).cuda()
    return out_imgs    

def noise_mask(noises, imgs, features = ['nose']):
    index = []
    for c in features:
        index.append(dict[c])
    batch_size = imgs.shape[0]
    noised_imgs = []
    for i in range(batch_size):
        img = imgs[i].unsqueeze(0)
        noise = noises[i].unsqueeze(0)
        with torch.no_grad():
            output = parsing_model(img)
            output = output[0][-1]
            output = F.interpolate(
                     output, (imgs.shape[2], imgs.shape[3]), mode='bilinear', align_corners=True)
            parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            #print(parsing.shape)
            for idx in index:    #specific class
                parsing[parsing == idx] = 1e6
            parsing[parsing != 1e6] = 0
            parsing[parsing != 0] = 1
            parsing = torch.from_numpy(parsing).unsqueeze(0).cuda()
            noised_img = parsing*noise + img
            #print("here",noised_img.shape)
        noised_imgs.append(noised_img.squeeze(0).cpu().numpy())
    noised_imgs = torch.tensor(noised_imgs)
    return noised_imgs 
