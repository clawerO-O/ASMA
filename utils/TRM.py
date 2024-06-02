import cv2
import math
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
from skimage import filters
from skimage.transform import resize
from skimage.morphology import disk
from .dataset import DFDCDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate_model', default='vgg19',
                        help='The substitute network eg. vgg19')
    parser.add_argument('--target_model', default='vgg19',
                        help='The target model eg. vgg19')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--val_dataset_name', default='imagenet',choices=['imagenet'],
                        help='The dataset to be used as test')

    parser.add_argument('--p_active', default=True, type=bool,
                        help='maximize the positive activation the conv layer')
    parser.add_argument('--p_rate', default=0.8, type=float,
                        help='positive proportion of conv layer used')
    parser.add_argument('--n_active', default=True, type=bool,
                        help='minimize the negative activation the conv layer')
    parser.add_argument('--n_rate', default=0.8, type=float,
                        help='negative proportion of conv layer used(deactivation)')

    parser.add_argument('--seed', default=123,
                        help='random seed')
    parser.add_argument('--lam', default=1,
                        help='the parameter of negative activation')
    parser.add_argument('--epsilon', default=10/255,
                        help='the infinite norm limitation of UAP')
    parser.add_argument('--delta_size', default=32,
                        help='the size of delta')
    parser.add_argument('--uap_lr', default=0.1,
                        help='the leraning rate of UAP')

    parser.add_argument('--prior', default='gauss',choices=['gauss','jigsaw',None],
                        help='the range prior of perturbations')
    parser.add_argument('--prior_batch', default=1,
                        help='the batch size of prior')
    parser.add_argument('--std', default=10,
                        help='initialize the standard deviation of gaussian noise')
    parser.add_argument('--fre', default=1,
                        help='initialize the frequency of jigsaw image')
    parser.add_argument('--uap_path', default=None,
                        help='the path of UAP')
    parser.add_argument('--gauss_t0', default=400,

                        help='the threshold to adjust the increasing rate of standard deviation(gauss)')
    parser.add_argument('--gauss_gamma', default=10,
                        help='the step size(gauss)')
    parser.add_argument('--jigsaw_t0', default=600,
                        help='the threshold to adjust the increasing rate of standard deviation(jigsaw)')
    parser.add_argument('--jigsaw_gamma', default=1,
                        help='the step size(jigsaw)')
    parser.add_argument('--jigsaw_end_iter', default=4200,
                        help='the iterations which stop the increment of frequency(jigsaw)')
    args = parser.parse_args()
    return args

def blur(img_temp, blur_p, blur_val):
    if blur_p > 0.5:
        return cv2.GaussianBlur(img_temp, (blur_val, blur_val), 1)
    else:
        return img_temp

def downsample(inp):
    return np.reshape(inp[1:-2, 1:-2, :], [1, 224, 224, 3])

def normalize(imgs):
    mean = torch.Tensor([0.5, 0.5, 0.5])#[0.485, 0.456, 0.406]
    std = torch.Tensor([0.5, 0.5, 0.5])#[0.229, 0.224, 0.225]
    return (imgs - mean.type_as(imgs)[None, :, None, None]) / std.type_as(imgs)[None, :, None, None]

def upsample(inp):
    out = np.zeros([227, 227, 3])
    out[1:-2, 1:-2, :] = inp
    out[0, 1:-2, :] = inp[0, :, :]
    out[-2, 1:-2, :] = inp[-1, :, :]
    out[-1, 1:-2, :] = inp[-1, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -2, :] = out[:, -3, :]
    out[:, -1, :] = out[:, -3, :]
    return np.reshape(out, [1, 227, 227, 3])

def flip(I, flip_p):
    if flip_p > 0.5:
        return I[:, ::-1, :]
    else:
        return I

def rotate(img_temp, rot, rot_p):
    if(rot_p > 0.5):
        rows, cols, ind = img_temp.shape
        h_pad = int(rows*abs(math.cos(rot/180.0*math.pi)) +
                    cols*abs(math.sin(rot/180.0*math.pi)))
        w_pad = int(cols*abs(math.cos(rot/180.0*math.pi)) +
                    rows*abs(math.sin(rot/180.0*math.pi)))
        final_img = np.zeros((h_pad, w_pad, 3))
        final_img[(h_pad-rows)//2:(h_pad+rows)//2, (w_pad-cols) //
                  2:(w_pad+cols)//2, :] = np.copy(img_temp)
        M = cv2.getRotationMatrix2D((w_pad//2, h_pad//2), rot, 1)
        final_img = cv2.warpAffine(
            final_img, M, (w_pad, h_pad), flags=cv2.INTER_NEAREST)
        part_denom = (math.cos(2*rot/180.0*math.pi))
        w_inside = int((cols*abs(math.cos(rot/180.0*math.pi)) -
                        rows*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        h_inside = int((rows*abs(math.cos(rot/180.0*math.pi)) -
                        cols*abs(math.sin(rot/180.0*math.pi)))/part_denom)
        final_img = final_img[(h_pad-h_inside)//2:(h_pad+h_inside)//2,
                              (w_pad - w_inside)//2:(w_pad + w_inside)//2, :].astype('uint8')
        return final_img
    else:
        return img_temp


def rand_crop(img_temp, dim=299):
    h = img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h = trig_w = False
    if(h > dim):
        h_p = int(random.uniform(0, 1)*(h-dim))
        img_temp = img_temp[h_p:h_p+dim, :, :]
    elif(h < dim):
        trig_h = True
    if(w > dim):
        w_p = int(random.uniform(0, 1)*(w-dim))
        img_temp = img_temp[:, w_p:w_p+dim, :]
    elif(w < dim):
        trig_w = True
    if(trig_h or trig_w):
        pad = np.zeros((dim, dim, 3), dtype='uint8')
        pad[:, :, 0] += 127
        pad[:, :, 1] += 127
        pad[:, :, 2] += 127
        pad[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        return pad
    else:
        return img_temp

def make_some_noise_gauss(std,size):
    '''
    The range prior for input with gauss noise
    '''
    mean = [127.5,127.5,127.5]
    sd = [std,std+10,std+20]
    im = np.zeros((size, size, 3))

    for i in range(3):
        im[:, :, i] = np.random.normal(
            loc=mean[i], scale=sd[i], size=(size, size))

    im = np.clip(im, 0, 255)
    return im

def randomizer(img_temp):
    dim = 224
    flip_p = random.uniform(0, 1)
    scale_p = random.uniform(0, 1)
    blur_p = random.uniform(0, 1)
    blur_val = random.choice([3, 5, 7, 9])
    rot_p = np.random.uniform(0, 1)
    rot = random.choice([-10, -7, -5, -3, 3, 5, 7, 10])
    if(scale_p > .5):
        scale = random.uniform(0.75, 1.5)
    else:
        scale = 1
    if(img_temp.shape[0] < img_temp.shape[1]):
        ratio = dim*scale/float(img_temp.shape[0])
    else:
        ratio = dim*scale/float(img_temp.shape[1])
    img_temp = cv2.resize(
        img_temp, (int(img_temp.shape[1]*ratio), int(img_temp.shape[0]*ratio)))
    img_temp = flip(img_temp, flip_p)
    img_temp = rotate(img_temp, rot, rot_p)
    img_temp = blur(img_temp, blur_p, blur_val)
    img_temp = rand_crop(img_temp)
    return img_temp

def img_preprocess(im,img_path=None, size=224, augment=False):
    '''
    A generic preprocessor for the range prior
    '''
    mean = [127.5,127.5,127.5]
    if img_path == None:
        img = im
    else:
        img = imread(img_path)
    if augment:
        img = randomizer(img)
    if len(img.shape) == 2:
        img = np.dstack([img, img, img])
    resFac = 256.0/min(img.shape[:2])
    newSize = list(map(int, (img.shape[0]*resFac, img.shape[1]*resFac)))
    img = resize(img, newSize, mode='constant', preserve_range=True)
    offset = [newSize[0]/2.0 -
              np.floor(size/2.0), newSize[1]/2.0-np.floor(size/2.0)]
    img = img[int(offset[0]):int(offset[0])+size,
              int(offset[1]):int(offset[1])+size, :]
    img[:, :, 0] -= mean[2]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[0]
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = np.reshape(img, [1, size, size, 3])
    return img

def curriculum_strategy_gauss(iter_num,args):
    if iter_num == 1600 or iter_num == 2400 or iter_num == 3200:  # vgg alexnet
        args.prior_batch = args.prior_batch * 2
    if iter_num % args.gauss_t0 == 0 and args.std < 127:
        args.std = args.std + args.gauss_gamma
    if args.std > 127:
        args.std = 127
    return args

def get_gauss_prior(args):
    '''
    The Gaussian noise is used
    as range-prior to simulate the real image.
    '''

    for i in range(args.prior_batch):
        im = None
        if args.prior == 'gauss':
            im = make_some_noise_gauss(args.std,args.delta_size)
        # elif args.range_prior == 'uniform':
        #     im = make_some_noise_uniform(args.std,args.delta_size)
        else:
            return None
            #im = make_some_noise_uniform(args.std)
            #im = make_cifar10_noise(args.std)
        # if prior_path == None and im == None:
        #     return None
        prior = img_preprocess(im = im,size=args.delta_size,augment=True)
        prior = np.moveaxis(prior, -1, 1)/255
        prior = torch.Tensor(prior)#.unsqueeze(0)
        if i == 0:
            prior_batch = prior
        else:
            prior_batch = torch.cat([prior_batch, prior], dim=0)


    return prior_batch

def get_conv_layers(model):
    '''
    Get all the convolution layers in the network.
    '''
    return [module for module in model.modules() if type(module) == nn.Conv2d]


def l2_layer_loss(model, delta,args,device):
    '''
    Compute the loss of TRM
    '''
    loss = torch.tensor(0.)
    activations = []
    p_activations = []
    deactivations = []
    remove_handles = []

    def check_zero(tensor):
        if tensor.equal(torch.zeros_like(tensor)):
            return False
        else:
            return True
    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None
    #print()
    #print('here')
    #print(len(get_conv_layers(model)))
    #print()
    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)

    model.eval()
    model.zero_grad()
    model(delta)

    # unregister hook so activation tensors have no references
    for handle in remove_handles:
        handle.remove()

    #calculation of truncated positive activation
    if args.p_active == True:
        # calculate the number of retained layers
        truncate = int(len(activations)* args.p_rate)
        if truncate <=0 and args.p_rate != 0.0:
            #avoid the zero of the number of the retained layer, i.e. truncated>=1
            truncate += 1
        for i in range(truncate):
            ac_tensor = activations[i].view(-1)
            # activate the positve value like Relu
            ac_tensor = torch.where(ac_tensor > 0, ac_tensor, torch.zeros_like(ac_tensor))
            p_activations.append(ac_tensor)

        truncate = int(len(activations)* args.n_rate)

        #calculation of truncated negative activation
        if truncate <=0 and args.n_rate != 0.0:
            truncate += 1
        for i in range(truncate):
            ac_tensor = activations[i].view(-1)
            # activate the negative value contrary to Relu
            ac_tensor = torch.where(ac_tensor < 0, ac_tensor, torch.zeros_like(ac_tensor))
            deactivations.append(ac_tensor)

    else:
        # maximize the positive activation of all layers
        for i in range(len(activations)):
            activations[i] = torch.where(activations[i] > 0, activations[i], torch.zeros_like(activations[i])).to(device)

    #calculate the loss by truncated ratio maximization problem
    if args.p_active == True:
        #add a tiny decement(1e-9) to avoid the zero value of activations
        p_loss = sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2+ 1e-9), p_activations)))
        loss = -p_loss
        n_loss = 0
        #compute the loss of the negative part
        if args.n_active == True:
            n_loss = sum(list(map(lambda deactivation: torch.log(torch.norm(deactivation,2)/2+1e-9 ), deactivations)))
            loss = args.lam * n_loss - p_loss
            # #observe the change of loss
            # if args.loss_show == True:
            #     print(f'positivie loss:{p_loss} negetive loss:{n_loss} loss:{loss}')

            return loss,p_loss,n_loss
        return loss,p_loss,n_loss

    else:
        #calculate the loss of maximizing the positive activation of all layers
        loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2 + 1e-9), activations)))

        return loss,loss,0.0

def get_rate_of_saturation(delta, xi):
    """
    Returns the proportion of pixels in delta
    that have reached the max-norm limit xi
    """
    return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)

def truncated_ratio_maximization(model, args):
    """
    Compute the UAP with the truncated ratio maximization.
    Return a single UAP tensor.
    """
    debug = False
    max_iter = 10000
    size = args.delta_size

    sat_threshold = 0.00001
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5
    sat_should_rescale = False

    iter_since_last_fooling = 0
    iter_since_last_best = 0
    best_fooling_rate = 0
    iter_num = 0

    xi_min = -10/255
    xi_max = 10/255
    args.std = 10


    delta = (xi_min - xi_max) * torch.rand((1, 3, size, size), device=device) + xi_max
    delta.requires_grad = True

    print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

    optimizer = optim.Adam([delta], lr=0.1)

    #val_loader,_ = get_data_loader(args.val_dataset_name, batch_size=args.batch_size)#,shuffle=True
    transform = A.Compose([
        A.Resize(299, 299),
        #A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    dataset = DFDCDataset(args.dataset_list, "val", args.val_dataset,
                              transform=transform, stable=True)
    val_loader = DataLoader(dataset=dataset,
                            batch_size=args.bs,
                            shuffle=True,
                            num_workers=args.workers,
                            pin_memory=True
                                )

    for i in tqdm(range(max_iter)):
        iter_num +=1
        iter_since_last_fooling += 1
        optimizer.zero_grad()

        # Sample artifical images from gaussian or jigsaw distribtuion
        #if prior != None:

        if args.prior == 'gauss':

            args = curriculum_strategy_gauss(iter_num, args)
            random_batch = get_gauss_prior(args=args)
             
            if random_batch!=None:
                example_prior = delta + random_batch.to(device)
            else:
                example_prior = delta
        #else:
        #    example_prior = delta


        loss,p_loss,n_loss= l2_layer_loss(model, example_prior,args,device)
        loss.backward()

        # args.loss_show = False
        # if iter_since_last_fooling %50 == 0:
        #     args.loss_show = True

        optimizer.step()

        # Clip the UAP to satisfy the restrain of the infinite norm
        with torch.no_grad():
            delta.clamp_(xi_min, xi_max)

        # Compute rate of saturation on a clamped UAP
        sat_prev = np.copy(sat)
        sat = get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
        sat_change = np.abs(sat - sat_prev)

        if sat_change < sat_threshold and sat > sat_min:
            if debug:
                print(f"Saturated delta in iter {i} with {sat} > {sat_min}\nChange in saturation: {sat_change} < {sat_threshold}\n")
            sat_should_rescale = True

        # fooling rate is measured every 200 iterations if saturation threshold is crossed
        # otherwise, fooling rate is measured every 400 iterations
        if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
            iter_since_last_fooling = 0

            print("\nGetting latest fooling rate...")

            current_fooling_rate = get_fooling_rate(model, torch.clamp(delta,xi_min,xi_max), val_loader, device)
            print(f"\nLatest fooling rate: {current_fooling_rate}")

            if current_fooling_rate > best_fooling_rate:
                print(f"Best fooling rate thus far: {current_fooling_rate}")
                best_fooling_rate = current_fooling_rate
                #best_uap = delta
            else:
                iter_since_last_best += 1

            # if the best fooling rate has not been overcome after patience_interval iterations
            # then training is considered complete
            if iter_since_last_best >= args.patience_interval:
                break

        if sat_should_rescale:
            #if the UAP is saturated, then compress it
            with torch.no_grad():
                delta.data = delta.data / 2
            sat_should_rescale = False
    delta = F.interpolate(
            delta, size=[299, 299], mode='bilinear')
    return delta

def get_fooling_rate(model, delta, data_loader, device):
    """
    Computes the fooling rate of the UAP on the dataset.
    """
    flipped = 0
    total = 0
    delta = F.interpolate(
               delta, size=[299, 299], mode='bilinear')
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(normalize(images))
            _, predicted = torch.max(outputs.data, 1)

            adv_images = torch.add(delta, images).clamp(0, 1)
            adv_outputs = model(normalize(adv_images))

            _, adv_predicted = torch.max(adv_outputs.data, 1)

            total += images.size(0)
            flipped += (predicted != adv_predicted).sum().item()

    return flipped / total
