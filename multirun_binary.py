import subprocess

# Class Names -----------------------------------------------------------------------------------------------------------
# acute_appendicitis, acute_cholecystitis, biliary_stone, abdominal_aortic_aneurysm, 
# active_bleeding, acute_diverticulitis, acute_pancreatitis, acute_pyelonephritis, 
# ureter_stone, mass_(suspicious_malignancy), aortic_dissection, hemoperitoneum, 
# abscess, bowel_obstruction, pneumoperitoneum, epiploic_appendagitis, 
# sma_lesion_(thrombosis,_dissection), celiac_lesion, hematoma, peptic_ulcer_disease, adrenal_lesion_(adenoma,_hyperplasia)
# ---------------------------------------------------------------------------------------------------------------------

# acute_class_name=$1 
# save_dir_name=$2


subprocess.run(['bash', 'train_binary_multiGPU.sh', 'acute_appendicitis', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'acute_cholecystitis', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'biliary_stone', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'abdominal_aortic_aneurysm', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'active_bleeding', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'acute_diverticulitis', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'acute_pancreatitis', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'acute_pyelonephritis', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'ureter_stone', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'mass_(suspicious_malignancy)', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'aortic_dissection', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'hemoperitoneum', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'abscess', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'bowel_obstruction', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'pneumoperitoneum', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'epiploic_appendagitis', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'sma_lesion_(thrombosis,_dissection)', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'celiac_lesion', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'hematoma', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'peptic_ulcer_disease', 'swint_v1_ours'])
subprocess.run(['bash', 'train_binary_multiGPU.sh', 'adrenal_lesion_(adenoma,_hyperplasia)', 'swint_v1_ours'])