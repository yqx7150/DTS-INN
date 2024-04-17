# Temporal Image Sequence Separation in Dual-tracer 
Dynamic PET with an Invertible Network



## The training pipeline of DTS-INN

 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig3.png"> </div>

## The detailed architecture of DTS-INN

 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig6.jpg"> </div>

## Visualization results of several comparison methods

 <div align="center"><img src="https://github.com/yqx7150/iVAN/blob/main/figs/Fig9.jpg"> </div>


# Train

Prepare your own datasets for DTS-INN

In the training process, you need to prepare at least two different tracers, as well as three datasets containing mixtures of two tracers. These datasets should be saved in different directories under ./data/data1/train/. You can train the model using the dataset flag --root1 './data/data1/train'. Optionally, you can create a hold-out test dataset at ./data/data1/test/ to evaluate your model.

##  

```python
python train.py 
```

##  resume training:

To fine-tune a pre-trained model, or resume the previous training, use the --resume flag




# Test

```python
python test.py    --ckpt="./exps/out_path/checkpoint/latest.pth"
```



  * # Acknowledgement

    The code is based on [yzxing87/Invertible-ISP](https://github.com/yzxing87/Invertible-ISP)


    ### Other Related Projects

     * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)             

      * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)    

     * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

      * Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2310.01885)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        

      * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)       

      * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   