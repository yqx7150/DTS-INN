# Temporal Image Sequence Separation in Dual-tracer Dynamic PET with an Invertible Network    
Chuanfu Sun, Bin Huang, Jie Sun, Yangfan Ni, Huafeng Liu, Qian Xia, Qiegen Liu, Wentao Zhu         
IEEE Transactions on Radiation and Plasma Medical Sciences     
https://ieeexplore.ieee.org/abstract/document/10542421       

Abstract:      
Positron emission tomography (PET) is a widely used functional imaging technique in clinic. Compared to single-tracer PET, dual-tracer dynamic PET allows two sequences of different nuclear pharmaceuticals in one scan, revealing richer physiological information. However, dynamically separating the mixed signals in dual-tracer PET is challenging due to identical energy ~511 keV in gamma photon pairs from both tracers. We propose a method for dynamic PET dual-tracer separation based on invertible neural networks (DTS-INNs). This network enables the forward and backward process simultaneously. Therefore, producing the mixed image sequences from the separation results through backward process may impose extra constraints for optimizing the network. The loss is composed of two components corresponding to the forward and backward propagation processes, which results in more accurate gradient computations and more stable gradient propagation with cycle consistency. We assess our model’s performance using simulated and real data, comparing it with several reputable dual-tracer separation methods. The results of DTS-INN outperform counterparts with lower-mean square error, higher-structural similarity, and peak signal to noise ratio. Additionally, it exhibits robustness against noise levels, phantoms, tracer combinations, and scanning protocols, offering a dependable solution for dual-tracer PET image separation.     

## The training pipeline of DTS-INN

 ![fig2](https://github.com/yqx7150/DTS-INN/assets/26964726/819935f9-5461-4c57-95ec-ea8e74c9e2f5)

## The detailed architecture of DTS-INN

 ![fig3](https://github.com/yqx7150/DTS-INN/assets/26964726/de4f60c8-a986-4d37-ae75-331943268fda)

## Visualization results of several comparison methods

 ![fig6](https://github.com/yqx7150/DTS-INN/assets/26964726/b33587b4-c313-4f9f-acb0-4846e3f42aa7)



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
python test.py --ckpt="./exps/out_path/checkpoint/latest.pth"
```



### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/algorithm-overview.png" width = "800" height = "500"> </div>
 Some examples of invertible and variable augmented network: IVNAC, VAN-ICC, iVAN and DTS-INN.          
           
     
  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)         
 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    
  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)        
  * Synthetic CT Generation via Variant Invertible Network for Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10666843)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)      
  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/IVPSQA/)    

  * Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10980366)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/st-DTPM)   
