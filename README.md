# random densenet

cd <to random_experiment> # ִ��·��
	# ִ������ο�
python3.6 random_main.py --model condensenet cifar10 --savedir random/Densenet --gpu 6 --range 1-5 --resume
# ����˵��
	--resume : ֧�ֶϵ�����һֱʹ�þͿ���
	--range 1-5: index Ϊ1 ��5 ��5��ʵ�飬��ֹ�����������ã��ܷ�Χ1-1100,�������ͬʱִ��ʱ���range��ʾ�ķ�Χ��Ҫ����
	--gpu 6�� ʹ��6��gpu
	--savedir: ����洢·��

# random resnet

cd <to random_experiment> # ִ��·��
	# ִ������ο�
python3.6 random_resnet.py cifar10 --savedir random/Resnet --gpu 6 --range 1-5 --resume
# ����˵��
	--resume : ֧�ֶϵ�����һֱʹ�þͿ���
	--range 1-5: index Ϊ1 ��5 ��5��ʵ�飬��ֹ�����������ã��ܷ�Χ1-1100,�������ͬʱִ��ʱ���range��ʾ�ķ�Χ��Ҫ����
	--gpu 6�� ʹ��6��gpu
	--savedir: ����洢·��