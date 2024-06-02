def gen_pic():
    image = Image.open('demo_.jpg').convert("RGB")
    transform = A.Compose([
        A.Resize(299, 299),
        #A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    out_dir = 'results/'
    #attack_model = 
    eval_model = xception(num_classes=2, pretrained=False)
    ckpt = torch.load("xceptionnet.pth", map_location="cpu")
    eval_model.load_state_dict(ckpt["state_dict"])
    eval_model.eval()
    eval_model = nn.DataParallel(eval_model)
    eval_model = eval_model.cuda()

    t_img = transform(image=np.array(image))["image"]/255
    img = t_img.unsqueeze(0).cuda()
    label = torch.LongTensor([1]).cuda()
    save_img(img*255, out_dir, tab="ori_")
    #FGSM
    attack = torchattacks.FGSM(eval_model, eps=8/255)
    adv_img = attack(img, label)
    save_img(adv_img*255, out_dir, tab="fgsm_", sec="fgsm attack")

    #BIM
    attack = torchattacks.BIM(eval_model, eps=8/255, alpha=2/255, steps=10)
    adv_img = attack(img, label)
    save_img(adv_img*255, out_dir, tab="bim_", sec="bim attack")

    #PGD
    attack = torchattacks.PGD(eval_model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    adv_img = attack(img, label)
    save_img(adv_img*255, out_dir, tab="pgd_", sec="pgd attack")

    #DeepFool
    attack = torchattacks.DeepFool(eval_model, steps=50, overshoot=0.02)
    adv_img = attack(img, label)
    save_img(adv_img*255, out_dir, tab="deepfool_", sec="deepfool attack")

    #C&W
    attack = torchattacks.CW(eval_model, c=1, kappa=0, steps=100, lr=0.01)
    adv_img = attack(img, label)
    save_img(adv_img*255, out_dir, tab="cw_", sec="cw attack")

    #ASMA
    adv_img = asma(img, label, eval_model)
    save_img(adv_img*255, out_dir, tab="asma_", sec="asma attack")

    #TSAA
    adv_img = tsaa(img)
    save_img(adv_img*255, out_dir, tab="tsaa_", sec="tsaa attack")

if __name__ == '__main__':
    gen_pic()
