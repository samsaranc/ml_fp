from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
SAVE_PROCESSED_IMGS = True

import numpy as np
from sklearn.preprocessing import normalize
import glob, os, time, copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import PUtils as P
from PNet import resnetP, PSampler

TRAINING_PHASE = ['tra', 'val']
RGBmean, RGBstdv = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EPOCH_MULT = 8

class learn():
	def __init__(self, src, dst, gpuid, batch_size=175, num_workers=20):
		self.src = src
		self.dst = dst
		self.gpuid = gpuid

		#whether we are processing on multiple GPUs or not
		if len(gpuid)>1:
			self.mp = True
		else:
			self.mp = False

		self.batch_size = batch_size
		self.num_workers = num_workers
		self.init_lr = 0.0001
		self.lr_decay_epoch = 4
		self.criterion = nn.CrossEntropyLoss()
		self.record = {p:[] for p in TRAINING_PHASE}

	def run(self):
		if not self.setsys(): print('system error');return
		self.loadData()
		self.setModel()
		self.printInfo()
		num_epochs=self.lr_decay_epoch*EPOCH_MULT
		self.train(num_epochs)

	def setsys(self):
		if not os.path.exists(self.src): print('src folder not exited'); return False
		if not torch.cuda.is_available(): print('No GPU detected'); return False
		if not os.path.exists(self.dst): os.makedirs(self.dst)
		torch.cuda.set_device(self.gpuid[0]); print('Current device is GPU: {}'.format(torch.cuda.current_device()))
		return True

	def loadData(self):
		data_transforms = {'tra': transforms.Compose([
								  transforms.Resize(size=300, interpolation=Image.BICUBIC),
								  transforms.RandomCrop(224),
								  transforms.RandomHorizontalFlip(),
								  transforms.ToTensor(),
								  transforms.Normalize(RGBmean, RGBstdv)]),
						   'val': transforms.Compose([
								  transforms.Resize(size=300, interpolation=Image.BICUBIC),
								  transforms.CenterCrop(224),
								  transforms.ToTensor(),
								  transforms.Normalize(RGBmean, RGBstdv)])}

		# tensors_to_plot = torch.from_numpy(imgs['bicubic_np'])
		# torchvision.utils.save_image(tensors_to_plot, 's3_processed\\')

		self.dsets = {p: datasets.ImageFolder(os.path.join(self.src, p), data_transforms[p]) for p in TRAINING_PHASE}
		self.class2indx = self.dsets['tra'].class_to_idx
		self.indx2class = {v: k for k,v in self.class2indx.items()}
		self.class_size = {p: {k: 0 for k in self.class2indx} for p in TRAINING_PHASE }# number of images in each class
		self.N_classes = len(self.class2indx)# total number of classes
		self.bookmark = {p:[] for p in TRAINING_PHASE}# index bookmark
		torch.save(self.indx2class, self.dst+'indx2class.pth')
		print(len(self.dsets['val']))
		# number of images in each class
		for phase in TRAINING_PHASE:
			for key in self.class2indx:
				filelist = [f for f in glob.glob(self.src + phase + '/' + key + '/' + '*.jpg')]
				self.class_size[phase][key] = len(filelist)
		print(self.dsets)
		# index bookmark
		print(sorted(self.indx2class))
		print(self.class_size)
		for phase in TRAINING_PHASE:
			print(phase)
			sta,end = 0,0
			for idx in sorted(self.indx2class):
				print("   " + str(idx))
				classkey = self.indx2class[idx]
				print("   " + str(classkey))
				end += self.class_size[phase][classkey]
				self.bookmark[phase].append((sta,end))
				print('--')
				print(idx)
				print(self.dsets[phase][sta][1])
				print(self.dsets[phase][end-1][1])
				try:
					print(self.dsets[phase][end][1])
				except:
					print('end')
				sta += self.class_size[phase][classkey]

		return

	def setModel(self):
		# create whole model
		Pmodel = resnetP(self.N_classes)
		for param in Pmodel.parameters():
			param.requires_grad = False
		for param in Pmodel.fc.parameters():
			param.requires_grad = True
		# parallel computing and opt setting
		if self.mp:
			print('Training on Multi-GPU')
			self.batch_size = self.batch_size*len(self.gpuid)
			self.model = torch.nn.DataParallel(Pmodel,device_ids=self.gpuid).cuda()#
			self.optimizer = optim.SGD(self.model.module.fc.parameters(), lr=0.01, momentum=0.9)
		else:
			print('Training on Single-GPU')
			self.model = Pmodel.cuda()
			self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.01, momentum=0.9)
		return

	def printInfo(self):
		print('\nimages size of each class\n'+'-'*50)
		print(self.class_size)
		print('\nindex to class\n'+'-'*50)
		print(self.indx2class)
		print('\nbookmark\n'+'-'*50)
		print(self.bookmark)
		return

	def DataLoaders(self):
		self.sampler = {TRAINING_PHASE[0]:PSampler(self.bookmark[TRAINING_PHASE[0]]),TRAINING_PHASE[1]:PSampler(self.bookmark[TRAINING_PHASE[1]],balance=False)}
		self.dataLoader = {p: torch.utils.data.DataLoader(self.dsets[p], batch_size=self.batch_size, sampler=self.sampler[p], num_workers=self.num_workers, drop_last = True) for p in TRAINING_PHASE}
		return

	def lr_scheduler(self, epoch):
		lr = self.init_lr * (0.1**(epoch // self.lr_decay_epoch))
		if epoch % self.lr_decay_epoch == 0: print('LR is set to {}'.format(lr))
		for param_group in self.optimizer.param_groups: param_group['lr'] = lr
		return

	def train(self, num_epochs):
		# recording time and epoch acc and best result
		since = time.time()
		self.best_tra = 0.0
		self.best_epoch = 0
		first_pass = True #is this the beginning?

		for epoch in range(num_epochs):
			self.DataLoaders()
			print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)

			for phase in TRAINING_PHASE:
				# recording the result
				accMat = np.zeros((self.N_classes,self.N_classes))
				running_loss = 0.0
				N_T, N_A = 0,0
				sum_a =0
				acc =0

				# Adjust the model for different phase
				if phase == 'tra':
					self.lr_scheduler(epoch)
					if self.mp:
						self.model.module.train(True)  # Set model to training mode
						if epoch < int(num_epochs*0.3): self.model.module.d_rate(0.1)
						elif epoch >= int(num_epochs*0.3) and epoch < int(num_epochs*0.6): self.model.module.d_rate(0.1)
						elif epoch >= int(num_epochs*0.6) and epoch < int(num_epochs*0.8): self.model.module.d_rate(0.05)
						elif epoch >= int(num_epochs*0.8): self.model.module.d_rate(0)

					if not self.mp:
						self.model.train(True)  # Set model to training mode
						if epoch < int(num_epochs*0.3): self.model.d_rate(0.1)
						elif epoch >= int(num_epochs*0.3) and epoch < int(num_epochs*0.6): self.model.d_rate(0.1)
						elif epoch >= int(num_epochs*0.6) and epoch < int(num_epochs*0.8): self.model.d_rate(0.05)
						elif epoch >= int(num_epochs*0.8): self.model.d_rate(0)

				if phase == 'val':
					if self.mp:
						self.model.module.train(False)  # Set model to evaluate mode
						self.model.module.d_rate(0)

					if not self.mp:
						self.model.train(False)  # Set model to evaluate mode
						self.model.d_rate(0)

				# iterate batch
				for data in self.dataLoader[phase]:
					# get the inputs
					inputs_bt, labels_bt = data #<class 'torch.FloatTensor'> <class 'torch.LongTensor'>
					print("save processed data")
					if SAVE_PROCESSED_IMGS:
						for img in inputs_bt:
							print(type(img))
							tensors_to_plot = torch.from_numpy(img)
							torchvision.utils.save_image(tensors_to_plot, 's3_processed\\')

					# zero the parameter gradients
					self.optimizer.zero_grad()
					# forward
					outputs = self.model(Variable(inputs_bt.cuda()))
					_, preds_bt = torch.max(outputs.data, 1)
					preds_bt = preds_bt.cpu().view(-1)

					# calsulate the loss
					loss = self.criterion(outputs, Variable(labels_bt.cuda()))

					# backward + optimize only if in training phase
					if phase == 'tra':
						loss.backward()
						self.optimizer.step()

					# statistics
					running_loss += loss.data.item() #data[0]
					N_T += torch.sum(preds_bt == labels_bt).item()

					for i in range(1,len(preds_bt)) :
						if preds_bt[i] == labels_bt[i]:
							sum_a += 1
		#   total += labels.size(0)
			#correct += (predicted == labels).sum().item()
			#print(type(preds_bt))
		   # print(type(labels_bt))
#                    print(preds_bt == labels_bt)
#                    print(preds_bt,' pred', labels_bt, 'true label')
					N_A += len(labels_bt)
					for i in range(len(labels_bt)): accMat[labels_bt[i],preds_bt[i]] += 1

				# record the performance
				mat = normalize(accMat.astype(np.float64),axis=1,norm='l1')
				if phase == 'tra' :
					figname = 'Training Accuracy'
				else :
					figname = 'Validation Accuracy'
				P.matrixPlot(mat,self.dst + 'epoch/',  figname + str(epoch))
				print('accuracy matrix')
				print(mat)
				print('sum_a',sum_a,'N_A',N_A)
				epoch_tra = np.trace(mat)
				epoch_loss = running_loss / N_A
				#acc = float(sum_a / N_A)
				ACC =float(float(sum_a)/float(N_A))
				epoch_acc = float(float(sum_a)/float(N_A))
				print('acc ',acc)

				self.record[phase].append((epoch, epoch_loss, epoch_acc))

				if type(epoch_loss) != float: epoch_loss = epoch_loss.item() #epoch_loss[0]
				print('{:5}:\n Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


				# deep copy the model
				if phase == 'val' and epoch_tra > self.best_tra and epoch > num_epochs/2:
						self.best_tra = epoch_tra
						self.best_epoch = epoch
						self.best_model = copy.deepcopy(self.model)
						torch.save(self.best_model, self.dst + 'model.pth')


			time_elapsed = time.time() - since
			print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
			print('Best val Acc: {:4f} in epoch: {}'.format(self.best_tra,self.best_epoch))
			torch.save(self.record, self.dst + str(self.best_epoch) + 'record.pth')
			P.recordPlot(self.record, self.dst)
			return

	def view(self):
		P.folderViewL(self.src,self.dst+'montage/')
