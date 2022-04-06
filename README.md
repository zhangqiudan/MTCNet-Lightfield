# A Multi-Task Collaborative Network for Light Field Salient Object Detection

## Training stage
1. Prepare the training data and change the data path
2. Download the initial model 
3. Train the model: python3 run_train.py --mode train
## Testing stage
1. Download the pretrained model
2. Download a example of testing data
3. Test the model: python3 test.py --mode test --sal_mode h --model ./models/mtcnet.pth



