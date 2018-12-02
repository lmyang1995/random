# random densenet

cd <to random_experiment> # 执行路径
	# 执行命令参考
block5：
python3.6 train_one_model.py  --normal-name '0101011214 1111111110 ' -F 48 --celln 3  --train_cifar10/block5_ --gpu 1 --resume -b 96 --id 0

# 参数说明
	--resume : 支持断点续，一直使用就可以
	-b batch_size
	--gpu 6： 使用6号gpu
	--savedir: 结果存储路劲
block4:
python3.6 train_one_model.py  --normal-name '11010102 15113102 ' -F 44 --celln 4  --train_cifar10/block4_ --gpu 1 --resume -b 96 --id 0
block3:
python3.6 train_one_model.py  --normal-name '010101 721111 ' -F 32 --celln 6  --train_cifar10/block3_ --gpu 1 --resume -b 96 --id 0
