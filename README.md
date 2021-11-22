# Classification of NIH ChestXray dataset (work in progress)

Implementation of a Deep Learning model based on a resnet architecture for classificating chest x-rays.
Behind this idea is the implementation of multiple functions for reading and creating genertors for train and test subsets, avoiding data leakage, normalicing the data, implementing a weighted loss for confronting the data inbalance and training a ResNet34 model.

Because of hardware limitations the dataset used is a portion of the full dataset, and the results obtained show overfiting in training. the project will be continued after the solution of this problem

Future implementations to consider are to use pretrained features to complement training and developing Grad Cam for visual explanations of the model decisions.

# Virtual enviroment instalation (Windows):

Create a virtual enviroment (named '.venv'):

        python -m venv .venv
        
Activate .venv
        
        .venv\Scripts\activate
        
Install the packages using pip

        pip install -r requirements.txt



# Dataset
The dataset used is the NIH ChestXray dataset, advaliable at:
https://nihcc.app.box.com/v/ChestXray-NIHCC

# References
NIH Chest X-ray dataset: https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf

ResNet architecture: https://ieeexplore.ieee.org/document/7780459

Grad Cam: https://arxiv.org/abs/1610.02391
