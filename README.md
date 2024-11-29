### ILOC

The relevant code for the paper "ILOC: Combining Ordered Interval Localization and Offset Calculation for Multimodal Depression Detection".

## Requirements

- Python >= 3.9
- PyTorch ==2.2.1+cu118
- Specific environmental requirements can be found in the file "requirements. txt"

## Train
Operation process:

- 1: Bimodal dataset SEARCH training code: folder 'src_SEARCH_5.87'
 
	
- 2: Trimodal dataset DAIC_woz training code: folder 'src_DAIC_3.85'


- 3: Trimodal dataset DAIC_woz training code: folder 'src_AVEC2014'	


## Dataset
  Obtain CMDC, DAIC_woz, SEARCH and AVEC_2014 datasets from official channels and put them into the datasets file.
  
  Specific operation tips can be found in the datasets file.
  
  - 1: AVEC2014数据集申请官网：http://avec2013-db.sspnet.eu/
  --	经常打不开，建议直接给作者发邮件询问Michel.Valstar@imperial.ac.uk
	
  - 2: DAIC-WOZ数据集申请官网：https://dcapswoz.ict.usc.edu/ 下载打印协议签署后发送到boberg@ict.usc.edu
  --	数据处理流程：https://github.com/LIU70KG/daic_woz_processing_master
 
  - 3: SEARCH数据集：该数据集来源于南京医科大学等多方机构合作进行的一项有关在校青少年心理健康长期变化追踪的研究。
	研究多阶段未完成，暂时未公开，后续将公开。公开后将进行更新申请链接。
  --	数据处理流程：dataset/SEARCH/SEARCH_process-master
 
  - 4: CMDC数据集申请官网：https://ieee-dataport.org/open-access/chinese-multimodal-depression-corpus
  --	官方提供数据集处理后的各模态特征 
  --	数据处理流程：dataset/CMDC/cmdc_data_pkl.py
