export CUDA_VISIBLE_DEVICES="0,1,2,3" 

root_dir=/mnt/BYS/github_codes/Gangnam_Sev
acute_class_name=$1 
save_dir_name=$2

# Class Names -----------------------------------------------------------------------------------------------------------
# acute_appendicitis, acute_cholecystitis, biliary_stone, abdominal_aortic_aneurysm, 
# active_bleeding, acute_diverticulitis, acute_pancreatitis, acute_pyelonephritis, 
# ureter_stone, mass_(suspicious_malignancy), aortic_dissection, hemoperitoneum, 
# abscess, bowel_obstruction, pneumoperitoneum, epiploic_appendagitis, 
# sma_lesion_(thrombosis,_dissection), celiac_lesion, hematoma, peptic_ulcer_disease, adrenal_lesion_(adenoma,_hyperplasia)
# ---------------------------------------------------------------------------------------------------------------------

# Model Names & Weights --------------------------------------------------------------------------------------------------
# swint:      /mai_nas/BYS/SSL/GBT/runs/MR_CT_100k_swin_modified/checkpoint100000.pth 
# swint_v2 :  /mai_nas/BYS/SSL/GBT/runs/MR_CT_100k_swin_unetr_v2/checkpoint100000.pth
# ---------------------------------------------------------------------------------------------------------------------

# if you want to use majority undersampling, add the argument --majority_undersampling

fold_num=0
python main.py \
    --model swint \
    --batch_size 2 \
    --task_name ${acute_class_name} \
    --logdir ${root_dir}/runs/${acute_class_name}/${save_dir_name}/fold_$fold_num \
    --fold_num $fold_num \
    --pretrained_checkpoint /mai_nas/BYS/SSL/GBT/runs/MR_CT_100k_swin_modified/checkpoint100000.pth \
    --distributed \
    --dist-url tcp://127.0.0.1:23434 \
    --use_ssl_pretrained True \
    --majority_undersampling \

export CUDA_VISIBLE_DEVICES="0" 
BEST_THRES=$(python eval.py \
    --model swint \
    --task_name ${acute_class_name} \
    --pretrained_dir ${root_dir}/runs/${acute_class_name}/${save_dir_name}/fold_$fold_num \
    --logdir /mai_nas/BYS/SSL/Finetune/Gangnam_Sev/runs/${acute_class_name}/${save_dir_name}/fold_$fold_num \
    --test_data_key validation \
    --pretrained_model_name model_best_auc.pt \
    --fold_num $fold_num \
    | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')
python eval.py \
    --model swint \
    --task_name ${acute_class_name} \
    --pretrained_dir ${root_dir}/runs/${acute_class_name}/${save_dir_name}/fold_$fold_num \
    --logdir /mai_nas/BYS/SSL/Finetune/Gangnam_Sev/runs/${acute_class_name}/${save_dir_name}/fold_$fold_num  \
    --test_data_key test \
    --pretrained_model_name model_best_auc.pt \
    --fold_num $fold_num \
    --best_thres $BEST_THRES