import subprocess

# acute_class_name=$1 
# save_dir_name=$2

subprocess.run(['bash', 'train_multiclass_multiGPU.sh', '4Class+Normal', 'swint_v1_ours'])
# subprocess.run(['bash', 'train_multiclass_multiGPU.sh', '6Class+Normal', 'swint_v1_ours'])