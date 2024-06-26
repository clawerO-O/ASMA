This is the code to reproduce our algorithm.

## Reproduce Guide

###Dataset Download
You can download the whole dataset we use in our experiment in `https://ai.facebook.com/datasets/dfdc`
You should unzip the whole dataset in `/dataset/videos/`

### Dataset Processing

The script `make_dataset.py` extracts aligned faces from a video file and save as images. It works like this:

```
$ mkdir /dataset/frames/
$ python make_dataset.py /dataset/video.mp4 /dataset/frames/
```

The script `make_dataset.sh` finds all mp4 files recursively in a directory and calls `make_dataset.py` on each mp4 file.

Supposing that you have downloaded DFDC datasets and extracted all zip files to `videos/`, run the following command to process the whole dataset and save face images to `/dataset/frames/` for training:

```
$ bash make_dataset.sh /dataset/videos/ /dataset/frames/
```

### Models Training

We simply use two-class models as per-face classifier.
We will use XceptionNet, Resnet-50, EfficientNet-b0, Efficient-b4 in our experiment.
For each model's training, check `xxx.conf` for its various settings, then you can run:
```
$ python train-xxx.py 
```
Also, you can tune it yourself.
The model you train will be saved in the folder with the same name of the model under the folder named pretrained.
For example, if you execute `$ python train-xception.py`, the pre-trained model parameters will be saved to the folder named xception.
The model parameters should be moved to the `checkpoints` file for testing.

### Pretrained Models
The training parameters for the models used in the experiments are available for download via `https://drive.google.com/drive/folders/1xmp3yMGYgY0xdjq5fvWIOzOOzsW2R-02?usp=sharing`.
Then You should save the models under `checkpoints` floder.

### Adversarial Attack Method
In our experiments, we have used some adversarial attack algorithms to compare with the proposed algorithm. 
| Method    |                                                                                                    Paper                                                                          |       Code       |
|   :---:         |                                                                                                     :----:                                                                           |        :---:        |
| FGSM        | Explaining and harnessing adversarial examples([Goodfellow et al., 2014])(https://arxiv.org/abs/1412.6572) | [Official Code](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) |
| BIM           | Adversarial Examples in the Physical World([Kurakin et al., 2016])(https://arxiv.org/abs/1607.02533)              | [Official Code](https://www.neuralception.com/adversarialexamples-bim) |
| PGD          | Towards Deep Learning Models Resistant to Adversarial Attacks([Mardry et al., 2017])(https://arxiv.org/abs/1706.06083) | [Official Code](https://github.com/lts4/deepfool) |
| DeepFool | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks([Moosavi-Dezfooli et al., 2016])(https://arxiv.org/abs/1511.04599) | [Official Code](https://github.com/lts4/deepfool) |
| C&W         | Towards Evaluating the Robustness of Neural Networks([Carlini et al., 2016])(https://arxiv.org/abs/1608.04644) | [Official Code](https://github.com/carlini/nn_robust_attacks)  |
| TRM          | TRM-UAP: Enhancing the Transferability of Data-Free Universal Adversarial Perturbation via Truncated Ratio Maximization([Liu et al., 2023])(https://openaccess.thecvf.com/content/ICCV2023/html/Liu_TRM-UAP_Enhancing_the_Transferability_of_Data-Free_Universal_Adversarial_Perturbation_via_ICCV_2023_paper.html) | [Official Code](https://github.com/RandolphCarter0/TRMUAP) |
### Experiment
#### Attack Success Rate Comparison
Before conducting the experiment, you should transfer the traned models to the folder checkpoints.

You can get the result simply by run the simple.py:
```
$ python main.py 
```
In simple.py you can change the attack model and the evaluate model.

If you use the command line to execute the program, you can use `--model` to set up the attack model, `--classifier` to set the test model, `--perturb-mode` to set the attack method, etc.
The optional parameters for `--model` and `--classifier` are four models, xception, resnet, efficiencient0 and efficiencient4.
Also you can use `--perturb-mode` to set the attack algorithm used in the experiment. 
After the program execution is complete, the accuracy of the attacked model and the attack success rate of the attacking algorithm are displayed.
If you want to save the adversarial examples, then you can add the image storage path after the parameter `--save`.
All functions and usage can be learned using `--help`.

#### Image Quality Assessment
You may want to observe the details of images you generated by models, you just put a 'demo.jpg' in this folder.
Then you should run 
```
$ python gen_adv_img.py 
```
to get the adversarial images in folder named results.
You can set the paths of the original image and the adversarial examples in iqa.py and execute 
```
$ python iqa.py 
```
to output the QA metric values of the adversarial samples generated by the corresponding attack algorithm.

Ablation Experiment
This algorithm proposes a way to restrict the attack region by semantic mask, our attack algorithm can be seen in merge_face.py, which can control whether to perform the restriction of the attack region or not by setting the logical value of the parameter `flag`.
or example, when `flag` is True, the restriction function of the region will be canceled, so that we can compare the comparison of the effect of the attack before and after the restriction of the region.