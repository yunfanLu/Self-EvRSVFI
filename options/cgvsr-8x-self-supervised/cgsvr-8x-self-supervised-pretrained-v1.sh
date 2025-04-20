export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="./":$PYTHONPATH

python egvfi/main.py \
  --yaml_file="./options/cgvsr-8x-self-supervised/cgsvr-8x-self-supervised-pretrained-v1.yaml" \
  --log_dir="./log/cgvsr-8x-self-supervised/cgsvr-8x-self-supervised-pretrained-v1/" \
  --alsologtostderr=True