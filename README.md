### ILOC

The relevant code for the paper "ILOC: Combining Ordered Interval Localization and Offset Calculation for Multimodal Depression Detection".

## Requirements

- Python >= 3.9
- PyTorch ==2.2.1+cu118
- Specific environmental requirements can be found in the file "requirements. txt"

## Train
Operation process:


- 1: Bimodal dataset SEARCH training code: folder 'src_SEARCH'
 
 
	
- 2: Trimodal dataset DAIC_woz training code: folder 'src_DAIC'

  	The pretrained network of ILOC in DAIC_woz can be downloaded at this  [link](https://drive.google.com/drive/folders/1JaaqT_auoMuO8K7VPjq-REX1E7ZBLiYV?usp=drive_link)



- 3:  Dataset:
  Obtain CMDC, DAIC_woz, SEARCH and AVEC_2014 datasets from official channels and put them into the datasets file.
  
  Specific operation tips can be found in the datasets file.

