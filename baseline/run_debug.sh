CUDA_VISIBLE_DEVICES=2 \
python  Train.py  \
--lr=1e-5  \
--batch-size=32  \
--log-interval=10 \
--snapshot-interval=10000  \
--tot-grad-updates=110000  \
--train-dir='../Resources/Datasets/IMAGENET/val' \
--valdn-dir='../Resources/Datasets/BSDS/BSR/BSDS500/data/images/test' \
--exp-code=E2 \
#--log-file=log.txt  \
