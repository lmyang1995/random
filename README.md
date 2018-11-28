# random densenet

cd <to random_experiment> # 执行路径
	# 执行命令参考
python3.6 random_main.py --model condensenet cifar10 --savedir random/Densenet --gpu 6 --range 1-5 --resume
# 参数说明
	--resume : 支持断点续，一直使用就可以
	--range 1-5: index 为1 到5 的5组实验，起止可以随意设置，总范围1-1100,多个程序同时执行时最好range表示的范围不要交叉
	--gpu 6： 使用6号gpu
	--savedir: 结果存储路劲

# random resnet

cd <to random_experiment> # 执行路径
	# 执行命令参考
python3.6 random_resnet.py cifar10 --savedir random/Resnet --gpu 6 --range 1-5 --resume
# 参数说明
	--resume : 支持断点续，一直使用就可以
	--range 1-5: index 为1 到5 的5组实验，起止可以随意设置，总范围1-1100,多个程序同时执行时最好range表示的范围不要交叉
	--gpu 6： 使用6号gpu
	--savedir: 结果存储路劲