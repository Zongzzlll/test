import os
import torchvision.transforms as transforms

class Config:
	def __init__(self):
		self.image_size = 448
		self.epochs = 25
		self.train_dir = '/cache/af2020cv-v5-formal/data/'
		self.train_label_path = '/cache/af2020cv-v5-formal/training.csv'
		self.val_label_path = '/cache/af2020cv-v5-formal/test.csv'
		self.result_path = '/cache/result.csv'
		self.swa_result_path = '/cache/swa_result.csv'
		self.batch_size = 32
		self.optimizer = 'sgd' #sgd adam sgd_gc
		if self.optimizer == 'sgd' or self.optimizer == 'sgd_gc':
			self.learn_rate = 0.01
		else:
			self.learn_rate = 0.00025
		self.scheduler = 'multi_step' #cosine, step, multi_step
		self.backbone = 'se_resnext101_32x4d' #'se_resnext101_32x4d' 'se_resnext50_32x4d'

		self.loss_fn = 'softmax'
		#self.loss_fn = 'focal'
		self.weight_decay = 5e-4
		self.warmup_iter = 200
		self.pooling = 'GeM'

		self.show_loop = 20
		self.save_dir = '/cache/checkpoints/'
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		## swa ###
		self.use_swa = True
		self.swa_start = 12
		self.swa_c_epochs = 1

		self.transform = transforms.Compose([
			transforms.Resize((self.image_size, self.image_size)),
			transforms.RandomAffine(10),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		self.test_transform = transforms.Compose([
			transforms.Resize((self.image_size, self.image_size)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])