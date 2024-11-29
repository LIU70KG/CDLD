### ILOC

The relevant code for the paper "ILOC: Combining Ordered Interval Localization and Offset Calculation for Multimodal Depression Detection".

## Requirements

- Python >= 3.9
- PyTorch ==2.2.1+cu118
Specific environmental requirements can be found in the file "requirements. txt"

## Train
Operation process:

- 1:Take the pretrained model xxx.pth file of CLLNet to the model folder
 
 	The pretrained network of ILOC in DAIC_woz can be downloaded at this  [link](https://drive.google.com/drive/folders/1JaaqT_auoMuO8K7VPjq-REX1E7ZBLiYV?usp=drive_link)
 
	
- 2:Take the pretrained model msceleb to the src folder
The pretrained network can be downloaded at this [link](https://drive.google.com/file/d/18oYDrZJnf4y9IkhSDZ6AO3nlRj_IvMAk/view?usp=sharing)


- 3:  Put the xxx.py file in the model folder to the src folder, and then run this file.

- 4:  Dataset:
  Obtain RAF-DB, AffectNet,SFEW and CAER-S datasets from official channels and put them into the datasets file.
  
  such as:
  Download basic emotions dataset of [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), and make sure it have a structure like following:
```
- datasets/raf-basic/
         EmoLabel/
             list_patition_label.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...
