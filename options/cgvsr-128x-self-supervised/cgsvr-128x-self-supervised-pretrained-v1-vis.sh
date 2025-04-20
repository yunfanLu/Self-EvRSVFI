export CUDA_VISIBLE_DEVICES="2"
export PYTHONPATH="./":$PYTHONPATH

python egvfi/main.py \
  --yaml_file="./options/cgvsr-128x-self-supervised/cgsvr-128x-self-supervised-pretrained-v1.yaml" \
  --log_dir="./log/cgvsr-128x-self-supervised/cgsvr-128x-self-supervised-pretrained-v1-test/" \
  --alsologtostderr=True \
  --VISUALIZE=True \
  --TEST_ONLY=True \
  --RESUME_PATH="/mnt/dev-ssd-8T/yunfanlu/workspace/iccv-23-vfi/01-eg-vfi/log/cgvsr-16x-self-supervised/cgsvr-16x-self-supervised-pretrained-v1/checkpoint-045.pth.tar"
