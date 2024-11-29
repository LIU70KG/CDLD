#!/bin/bash
# Define the hyperparameter combinations you want to try
#best_model_Configuration_Log_choice=("./MISA_backbone_report.txt")
best_model_Configuration_Log_choice=("./fragment_MISA_backbone.txt")
model_choice=('Simple_Fusion_Network' 'MISA_CMDC')
class_weight_choice=(10.0 5.0 3.0 2.0 1.0)
shifting_weight_choice=(2.0 1.5 1.0 0.0)
order_center_weight_choice=(2.0 1.5 1.0 0.0)
ce_loss_weight_choice=(1.0 2.0 3.0)
#pred_center_score_weight_choice=(0.0 0.05 0.1)

# Loop over hyperparameter combinations
for model in "${model_choice[@]}"; do
    for class_weight in "${class_weight_choice[@]}"; do
        for shifting_weight in "${shifting_weight_choice[@]}"; do
            for order_center_weight in "${order_center_weight_choice[@]}"; do
                for ce_loss_weight in "${ce_loss_weight_choice[@]}"; do
                    echo "-----Training+Testing with class_weight=$class_weight, shifting_weight=$shifting_weight, order_center_weight=$order_center_weight, ce_loss_weight=$ce_loss_weight, model=$model --------"
                    CUDA_VISIBLE_DEVICES=0 python train_1.py --class_weight $class_weight --shifting_weight $shifting_weight --order_center_weight $order_center_weight --ce_loss_weight $ce_loss_weight --model $model
                done
            done
        done
    done
done