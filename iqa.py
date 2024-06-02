from skimage.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np

ori_img = cv2.imread("ori.png")
adv_path = "ours(W)_resnet.png"
adv_img = cv2.imread(adv_path)
mse = compare_mse(ori_img, adv_img)
mae = np.mean(np.abs(ori_img/255-adv_img/255))
psnr = compare_psnr(ori_img, adv_img)
ssim = compare_ssim(ori_img, adv_img, multichannel=True)
print('********************************')
print(adv_path.split('.')[0])
print('MSE:',mse)
print('MAE:',mae)
print('PSNR:',psnr)
print('SSIM:',ssim)
print('********************************')
#face nose eye 
