CUDA_VISIBLE_DEVICES=2 \
python Train.py  \
--lr1=1e-4  \
--lr2=1e-4  \
--batch-size=32  \
--log-interval=10 \
--snapshot-interval=10000  \
--tot-grad-updates=110000  \
--train-dir='../Resources/Datasets/IMAGENET/val' \
--valdn-dir='../Resources/Datasets/BSDS/BSR/BSDS500/data/images/test' \
--exp-code=E2a \
--log-file=log.txt  \
