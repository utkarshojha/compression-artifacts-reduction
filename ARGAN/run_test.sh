CUDA_VISIBLE_DEVICES=1 \
python Test.py \
--batch-size 32 \
--stride 16 \
--quality 20 \
--model-snapshot-file  E2a/checkpoints/E2a.40000.ckpt \
--image-dir='../Resources/Datasets/LIVE1/' \
--test-data-list-file ../Resources/live1_list.txt  \
--save-image True \
