import os
os.environ["CUDA_VISBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from unet import unet
from FaceParse import FaceParseNet50
from PIL import Image
import numpy as np
import cv2 as cv
import attacks
#import torchattacks as attack
from model_def import xception

def merge(imgs, attack, features = ['nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow']):
    pretrained_path = '38_G.pth'
    parsing_model = FaceParseNet50(num_classes=19,pretrained=False)#.cuda()
    parsing_model.load_state_dict(torch.load(pretrained_path))
    parsing_model.eval()
    index = []
    for c in features:
        index.append(dict[c]) 
    batch_size = imgs.shape[0]
    indexs = []
    merged_imgs = []
    for i in range(batch_size):
        img = imgs[i].detach().unsqueeze(0)#.cuda()#(1,3,244,244)
        with torch.no_grad():
            img_ = F.interpolate(
                     img, (512, 512), mode='bilinear', align_corners=True)
            output = parsing_model(img_)
            output = output[0][-1]
            output = F.interpolate(
                     output, (imgs.shape[2], imgs.shape[3]), mode='bilinear', align_corners=True)
            parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)#(244,244)
            #print(parsing.shape)
            for idx in index:    #specific class 
                parsing[parsing == idx] = 1e6
            index = np.where(parsing==1e6)
            parsing[parsing != 1e6] = 0
            parsing[parsing == 1e6] = 1 #mark the targeted part 
            #parsing = torch.from_numpy(parsing).unsqueeze(0).numpy()#(1,244,244)    
            #print(parsing)
        merged_imgs.append([parsing, parsing, parsing])
        mask = torch.from_numpy(np.array(merged_imgs).astype(np.float32).squeeze(0))
        indexs.append(index)
 
    
    #return (torch.tensor(merged_imgs, dtype=torch.float32, device=torch.device('cuda:0'))-mean)/std
    #print(mask.shape)
    #merged_imgs = (merged_imgs - mean) / std #normalization to offset attack's effect
    '''  
    trans = transforms.Compose([
        #transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage(),
    ])
    mask = trans(mask)
    mask.save("mask.png")
 '''
    adv_mask = attack.perturb(imgs*std+mean, None)#adversarial perturb
    
# * torch.tensor(merged_imgs, dtype=torch.float32, device=torch.device('cuda:0'))#,0,1)
    #return (adv_mask-mean)/std
    #print(adv_mask)
    #adv_mask = adv_mask * std + mean #the same reason
    #print(imgs.shape, adv_mask.shape)
    out_imgs = []
    #out_imgs = imgs*std+mean + adv_mask-mean
    #return out_imgs/std
    for i in range(batch_size):   
        img = imgs[i].detach()
        #print(img.shape)
        img = (img * std + mean)
        #print(img)
        mask_ = adv_mask[i].detach()
        #print(mask)
        #print(img.shape,mask.shape)
        out_img = img.clone().detach()
        index = indexs[i]
        #print(index[0])
        out_img[:, index[0], index[1]] = mask_[:, index[0], index[1]]# + img[:, index[0], index[1]]  
        #out_img = img + mask
        #print("here",out_img)
        #print(out_img)
        out_img = (out_img - mean) / std  
        out_imgs.append(out_img.cpu().numpy())
    out_imgs = torch.tensor(out_imgs)#.cuda()
    return out_imgs 

dict = {'background':0, 'skin':1, 'nose':2, 'eye_g':3, 'l_eye':4, 'r_eye':5,
        'l_brow':6, 'r_brow':7, 'l_ear':8, 'r_ear':9, 'mouth':10, 'u_lip':11,
        'l_lip':12, 'hair':13, 'hat':14, 'ear_r':15, 'neck_l':16, 'neck':17,
        'cloth':18}


#key = 'hair'
key = 'nose'
#print(dict[key])

def vis_parsing_maps(im, parsing_anno, color=[230, 50, 20]):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],[255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153],[0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros(
        (parsing_anno.shape[0], parsing_anno.shape[1], 3))

    for pi in range(len(part_colors)):
        index = np.where(parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv.addWeighted(cv.cvtColor(
        vis_im, cv.COLOR_RGB2BGR), 1.0, vis_parsing_anno_color, 0.5, 0)
    vis_im = Image.fromarray(cv.cvtColor(vis_im, cv.COLOR_BGR2RGB))

    return vis_im
def gen_mask(parsing_anno):
    
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    face_parsing = (parsing_anno == dict[key])
    mask[:, :, 0] = np.where(face_parsing, 0, 0)
    mask[:, :, 1] = np.where(face_parsing, 0, 0)
    mask[:, :, 2] = np.where(face_parsing, 0, 0)
    mask = Image.fromarray(cv.cvtColor(mask, cv.COLOR_BGR2RGB))
    return mask
    #kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    #mask = cv.erode(mask, kernel)
    #contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(im, contours, -1, (255, 255, 0), 5)
    #cv.imwrite("mask.png", mask[:, :, ::-1])
    
def mf(img,parsing):
    print("here",img[0].shape)
    #img = np.array(img)#.astype(np.float32)/255
    img = img.cpu().squeeze(0).permute(1,2,0).numpy()
    print(img.shape,parsing.shape)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   
    mask = np.zeros(
        (parsing.shape[0], parsing.shape[1], 3),dtype = np.float32)
    
    face_parsing = (parsing == dict[key])
      
    #print(face_parsing.shape)
    #for item in face_parsing:
    #print(face_parsing[0].shape)
    mask[:, :, 0] = np.where(face_parsing, 1, 0)
    mask[:, :, 1] = np.where(face_parsing, 1, 0)
    mask[:, :, 2] = np.where(face_parsing, 1, 0)
    print(mask.shape)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB) 
    vis_im = cv.bitwise_xor(img,mask)
    print(vis_im)
    #mask = mask.astype(np.uint8)
    #vis_im = cv.addWeighted(cv.cvtColor(
    #    vis_im, cv.COLOR_RGB2BGR), 1.0, mask, 0.5, 0)
    #vis_im = Image.fromarray(cv.cvtColor(vis_im, cv.COLOR_BGR2RGB)) 
    return vis_im

img_path ='1612.jpg'
pretrained_path = '38_G.pth'
pretrained_path1 = 'model.pth'
img = Image.open(img_path).convert("RGB")
img = img.resize((512, 512), Image.ANTIALIAS)
#print(np.array(img).shape)
transform = transforms.Compose([
    #transforms.Resize(244),
    transforms.ToTensor(),
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
mean, std = torch.tensor(mean).reshape(3, 1, 1), torch.tensor(std).reshape(3, 1, 1)#.cuda()
if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = torch.device('cpu')
print(device)
cudnn.benchmark = True    

net = FaceParseNet50(num_classes=19,pretrained=False).cuda()
net1 = unet()#.cuda()
net.load_state_dict(torch.load(pretrained_path))
net1.load_state_dict(torch.load(pretrained_path1))
net.eval()
net1.eval()
model = xception(num_classes=2, pretrained=None).cuda()
ckpt = torch.load('xceptionnet.pth', map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
img_ = transform(img).unsqueeze(0).to(device)
#ca = attacks.CAFA(model, eps=8/255, alpha=0.01, steps=100, random_start=True)
#fgsm = attacks.FGSM(model, eps=8/255)
#fgsm.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#ifgsm = attacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
#ifgsm.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#pgd = attacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
#pgd.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#cw = attack.CW(model, c=1, kappa=1, steps=100, lr=0.01)
#cw.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#df = attack.DeepFool(model, steps=50, overshoot=0.02)
#df.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#print(img_.shape)
'''
with torch.no_grad():
    output = net(img_)
    output = output[0][-1] 
    output = F.interpolate(
        output, (512, 512), mode='bilinear', align_corners=True)  # [1, 19, 512, 512]
    
    #parsing = torch.sgn(output - torch.max(output,1)[0].squeeze(1))+1 #[1,19,512,512]
    parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
    #print(parsing.shape) 
    #parsing = parsing.squeeze(0)
    #parsing = parsing[dict[key]].cpu().numpy()
    #print(parsing.shape)
    #out_img = gen_mask(parsing_anno=parsing)
    
    #out_img = merge(img_, ca)

    #print(parsing.shape)cv.cvtColor(out, cv.COLOR_BGR2RGB)

    out_img = vis_parsing_maps(img.resize((512,512)), parsing)
    out_img.resize((224,224))
    out_img.save("img_mask_colored.png")
#out = merge(img_,ca)
#out = cw(img_, torch.tensor([1]))
#out = ((out*std+mean)*255).round().squeeze(0).permute(1,2,0).cpu().numpy()
#out = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
#out.save('cw.png')'''
