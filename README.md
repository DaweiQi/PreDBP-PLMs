# PreDBP-PLMs:Prediction of DNA-binding proteins based on pre-trained protein language models and convolutional neural networks

## 1.Introduction
We proposed a novel predictor named PreDBP-PLMs to further improve the identification accuracy of DBPs by fusing the pre-trained
protein language model (PLM) ProtT5 embedding with evolutionary features as input to the classic convolutional neural network (CNN)
model.Compared to the existing state-of-the=art predictors,PreDBP-PLMs exhibits an accuracy improvement of 0.5% and 5.2% on the
PDB186 and PDB2272 datasets,respectively.It demonstrated that the proposed method could serve as a useful tool for the recognition of DBPs.

## 2.Requirements
numpy==1.20.3 <br>
pandas==1.5.3 <br>
scikit_learn==1.3.2 <br>
torch==1.9.0+cu102 <br>
torchvision==0.9.1+cu102 <br>

## 3.Usage
run ```Code/get_ProtT5_basic.py``` to generate pre-trained feature files.<br>
run ```Code/DBPs_independent_test_result.py``` to obtain the results of the 5-fold CV and independent test set of the model.
