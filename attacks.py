from collections import OrderedDict
import advertorch.utils as au
import numpy as np
import torch
from advertorch.attacks.utils import rand_init_delta
from torch import nn, optim
from advertorch.attacks import LabelMixin, Attack as A

def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get('_attacks').values():
            eval("atk."+func.__name__+"(*args, **kwargs)")
        return result
    return wrapper_func

class Attack(object):
    def __init__(self, name, model):
        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model)
        self.device = next(model.parameters()).device

        # Controls attack mode.
        self.attack_mode = 'default'

        # Controls when normalization is used.
        self.normalization_used = {}
        self._normalization_applied = False

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_normalization_used(self, mean, std):
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used['mean'] = mean
        self.normalization_used['std'] = std
        self._normalization_applied = True

    def normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs * std + mean

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError

    def _check_inputs(self, images):
        tol = 1e-4
        if self._normalization_applied:
            images = self.inverse_normalize(images)
        if torch.max(images) > 1 + tol or torch.min(images) < 0 - tol:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(images), torch.min(images)))
        return images

    def _check_outputs(self, images):
        if self._normalization_applied:
            images = self.normalize(images)
        return images

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits



    @wrapper_method
    def set_device(self, device):
        self.device = device


    def get_mode(self):
        return  self.attack_mode

    def __call__(self, images, labels=None, *args, **kwargs):
        images = self._check_inputs(images)
        adv_images = self.forward(images, labels, *args, **kwargs) if labels is not None \
                else self.forward(images, *args, **kwargs)
        adv_images = self._check_outputs(adv_images)
        return adv_images

    def __repr__(self):
        info = self.__dict__.copy()
        del_keys = ['model', 'attack', 'supported_mode']
        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)
        for key in del_keys:
            del info[key]
        info['attack_mode'] = self.attack_mode
        info['normalization_used'] = True if len(self.normalization_used) > 0 else False  # nopep8
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        attacks = self.__dict__.get('_attacks')
        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if (items not in stack):
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = (list(items.keys()) + list(items.values()))
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items
        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get('_attacks').items():
                attacks[name + "." + subname] = subvalue

class FGSM(Attack):

    """
        FGSM in the paper 'Explaining and harnessing adversarial examples'
        [https://arxiv.org/abs/1412.6572]

        Distance Measure : Linf
    """

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.get_logits(images)
        # Calculate loss
        cost = loss(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images

class BIM(Attack):

    """
        BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
        [https://arxiv.org/abs/1607.02533]

        Distance Measure : Linf
    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        ori_images = images.clone().detach()
        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.get_logits(images)
            # Calculate loss
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            adv_images = images + self.alpha * grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (adv_images < a).float() * a  # nopep8
            c = (b > ori_images + self.eps).float() * (ori_images + self.eps) + (
                        b <= ori_images + self.eps).float() * b  # nopep8
            images = torch.clamp(c, max=1).detach()
        return images

class PGD(Attack):

    """
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]

        Distance Measure : Linf
    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                         torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            # Calculate loss
            cost = loss(outputs, labels)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images

class CW(Attack):

    """
        CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
        [https://arxiv.org/abs/1608.04644]

        Distance Measure : L2
    """

    def __init__(self, model, c=1, kappa=0, steps=50, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        optimizer = optim.Adam([w], lr=self.lr)
        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)
            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.get_logits(adv_images)
            f_loss = self.f(outputs, labels).sum()
            cost = L2_loss + self.c * f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            condition = (pre != labels).float()
            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images
            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]
        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]
        return torch.clamp((real - other), min=-self.kappa)

class DeepFool(Attack):

    """
        'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
        [https://arxiv.org/abs/1511.04599]

        Distance Measure : L2
    """

    def __init__(self, model, steps=50, overshoot=0.02):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot

    def forward(self, images, labels):
        adv_images, target_labels = self.forward_return_target_labels(images, labels)
        return adv_images

    def forward_return_target_labels(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0
        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx + 1].clone().detach()
            adv_images.append(image)
        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1
        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.get_logits(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)
        ws = self._construct_jacobian(fs, image)
        image = image.detach()
        f_0 = fs[label]
        w_0 = ws[label]
        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]
        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)
        delta = (torch.abs(f_prime[hat_L]) * w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2) ** 2))
        target_label = hat_L if hat_L < label else hat_L + 1
        adv_image = image + (1 + self.overshoot) * delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)

class CAFA(Attack):

    """
        CAFA in paper 'Removing Adversarial Noise in Class Activation Feature Space'
        [https://arxiv.org/abs/2104.09197]

        Distance Measure : Linf
    """

    def __init__(self, model, eps=8/255, alpha=0.01, steps=10, random_start=True):
        super().__init__("CAFA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.layer_name = self.get_last_conv_name(self.model)

    def forward(self, images):
        images = images.clone().detach().to(self.device)
        images.requires_grad=True
        cam = CAM_tensor(self.model, self.layer_name)
        loss_fn = cam_criteria(cam).to(self.device)
        attack = LinfCAMAttack(
            [self.model], loss_fn=loss_fn, eps=self.eps,
            nb_iter=self.steps, eps_iter=self.alpha,
            rand_init=self.random_start,
            clip_max=1.0, clip_min=0.0, targeted=False
        )
        adv_images = attack.perturb(images,None).detach()
        return adv_images


    def get_last_conv_name(self, model):

        layer_name = None
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        return layer_name


def perturb_iterative(x_var, y_var, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param x_var: input data.
    :param y_var: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(x_var)

    delta.requires_grad_()
    for ii in range(nb_iter):
        # outputs = predict(xvar + delta)
        loss = loss_fn(x_var + delta, x_var)
        # print('loss:' + str(loss.item()), flush=True)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + au.batch_multiply(eps_iter, grad_sign)
            delta.data = au.batch_clamp(eps, delta.data)
            delta.data = au.clamp(x_var.data + delta.data, clip_min, clip_max
                               ) - x_var.data

        elif ord == 2:
            grad = delta.grad.data
            grad = au.normalize_by_pnorm(grad)
            delta.data = delta.data + au.batch_multiply(eps_iter, grad)
            delta.data = au.clamp(x_var.data + delta.data, clip_min, clip_max
                               ) - x_var.data
            if eps is not None:
                delta.data = au.clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = au.normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + au.batch_multiply(eps_iter, grad)

            delta.data = au.batch_l1_proj(delta.data.cpu(), eps)
            if x_var.is_cuda:
                delta.data = delta.data.cuda()
            delta.data = au.clamp(x_var.data + delta.data, clip_min, clip_max
                               ) - x_var.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = au.clamp(x_var + delta, clip_min, clip_max)
    return x_adv


class CAMAttack(A, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.

        """
        super(CAMAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert au.is_float_or_torch_tensor(self.eps_iter)
        assert au.is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = au.clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data

class LinfCAMAttack(CAMAttack):
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfCAMAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)

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

