export CUDA_VISIBLE_DEVICES=0
iter_number=1121
bs=124
#iter_number = 11200
#python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size 64
echo "-------------------------------------------finish"
#python3 train.py --name CRC --dataset CRC --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size 128 --cv 1 --eval_batch_size 128  \
	#--num_steps $((11200 * 10 )) \
	#--eval_every 11200
python3 -u train.py --name ECRC --dataset ECRC --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size $bs --cv 1 --eval_batch_size $bs  \
	--num_steps $(($iter_number * 10 )) \
	--eval_every $iter_number

echo "-------------------------------------------finish"
#python3 train.py --name CRC --dataset CRC --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size 128 --cv 2  --eval_batch_size 128  \
	#--num_steps $((11200 * 10 )) \
	#--eval_every 11200
python3 -u train.py --name ECRC --dataset ECRC --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size $bs --cv 2 --eval_batch_size $bs  \
	--num_steps $(($iter_number * 10 )) \
	--eval_every $iter_number

echo "-------------------------------------------finish"
#echo ''
#python3 train.py --name CRC --dataset CRC --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size 128 --cv 3  --eval_batch_size 128  \
#	--num_steps $((11200 * 10 )) \
#	--eval_every 11200
python3 -u train.py --name ECRC --dataset ECRC --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  --train_batch_size $bs --cv 3  --eval_batch_size $bs \
	--num_steps $(($iter_number * 10 )) \
	--eval_every $iter_number
