export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="./":$PYTHONPATH

python egvfi/main.py \
  --yaml_file="./options/aplex-vfi-dataset/cgsvr-32x-aplex-self-wo_df_loss-1224-v2.yaml" \
  --log_dir="./log/aplex-vfi-dataset/cgsvr-32x-aplex-self-wo_df_loss-1224-v2/" \
  --alsologtostderr=True