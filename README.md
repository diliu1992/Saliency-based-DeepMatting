# Saliency-based-DeepMatting

We proposed an automatic alpha prediction method. We use salient object detection method to generate initial object boundaries. Then dilation processing is adopted to generate the unknown area along the edge of image. The trimap is thus generated and been combined with the RGB to predict alpha matte using the method Deep Image Matting

## Step 1: prepare dataset

We provide three dataset to test our method, please download these datasets into the `dataset` folder outside the `DeepMatting` folder. The `get_dataset_loader.py` is for data loading.

- ***Adobe dataset***

    Follow the instruction to contact author for the dataset.
    
    <https://github.com/foamliu/Deep-Image-Matting>
    
    Go to MSCOCO to download:
    
    <http://images.cocodataset.org/zips/train2014.zip>
    
    Go to PASCAL VOC to download:
    
    <http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar>
    
    Download VGG16 into models folder.
    
    <https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5>
    
- ***Portrait dataset***
    
    <http://www.cse.cuhk.edu.hk/~leojia/projects/automatting/>
    
 - ***HumanHalf dataset***
    
    <https://github.com/aisegmentcn/matting_human_datasets>


## Step 2: build network

- ***Matting network: M-Net***

    The T-Net plays the role of salient object segmentation. I use RefineNet as T-Net to predict trimap in `./models/RefineNet.py`

- ***Matting network: M-Net***

    The M-Net aims to capture detail information and generate alpha matte. I use Deep image matting as M-Net to predict alpha matte in `./models/py_encoder_decoder.py`
  
## Step 3: build loss 

We use the `F.binary_cross_entropy` loss for stage 0: T_Net training, 

We use following loss function in `./utils/loss.py` for stage 1: M_Net training

```
    # -------------------------------------
    # matting prediction loss loss_matting
    # mask: mask for the unknown area of trimap
    # ------------------------
    eps = 1e-6
    # loss_alpha
    loss_alpha = torch.mul(torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps), mask).mean()

    # loss_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img
    loss_composition = torch.mul(torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps), mask).mean()

    loss_matting = 0.5 * loss_alpha + 0.5 * loss_composition
```

## Step 4: train

Select dataset `args['dataset']` in `'Adobe'`, `'Portrait'`, `'HumanHalf'`

Firstly, train T-Net, use ```./train.py``` with `args['stage']=0`

Then, train M-Net, use ```./train.py``` with `args['stage']=1`

# Test
  
  run ```./test.py``` to test on the Portrait dataset
  
  run ```./test_singel_image.py``` to select single image in ```./data/``` for testing, the result is writed into the ```./test/``` folder




