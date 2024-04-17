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

     

      * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)       

      * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
