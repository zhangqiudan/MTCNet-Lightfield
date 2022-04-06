# A Multi-Task Collaborative Network for Light Field Salient Object Detection

## Training stage
1. Prepare the training data and change the data path
2. Download the [initial model](https://pan.baidu.com/s/1cTkx2dbbyeT1jLSC1Uejfw) - hqpw to the weights file
3. Train the model: python3 run_train.py --mode train
## Testing stage
1. Download the [pretrained model](https://pan.baidu.com/s/13RvcOWeR3NaAs5vY2tyRRg)- rrg7
2. Download a example of [testing data](https://pan.baidu.com/s/15FrJ24Vq16yfYgjxm2FG5Q) -9249
3. Test the model: python3 test.py --mode test --sal_mode h --model ./models/mtcnet.pth





Our code is rewritten on the basis of this paper. Thanks a lot for the excellent code provided by the authors.

@inproceedings{zhao2019EGNet,
 title={EGNet:Edge Guidance Network for Salient Object Detection},
 author={Zhao, Jia-Xing and Liu, Jiang-Jiang and Fan, Deng-Ping and Cao, Yang and Yang, Jufeng and Cheng, Ming-Ming},
 booktitle={The IEEE International Conference on Computer Vision (ICCV)},
 month={Oct},
 year={2019},
}



