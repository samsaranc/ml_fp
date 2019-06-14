import pymysql
import mysql.connector
import os

from __future__ import print_function
import os, copy
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import datetime as dt
from statistics import mean
import csv

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from PUtils import invDict, matrixPlot
from PLoader import ImageReader

PHASE = ['train','val']
AWNS = ['AWNED','AWNLESS ']
# normalization paras
RGBmean, RGBstdv = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

#create connection
dbServerName    = "127.0.0.1"

dbUser          = "root"

dbPassword      = "ab0utT0D4Y_"

dbName          = "sd_research"

charSet         = "utf8mb4"

table           = "images"

cursorType     = pymysql.cursors.DictCursor

connectionObject   = mysql.connector.connect(
                        host=dbServerName,
                        user=dbUser,
                        password=dbPassword,
                        db=dbName,
                        auth_plugin='mysql_native_password'
                    ) #, cursorclass=cursorType

def checkfile(path):
	try:
		img = default_loader(path)
	except OSError:
		if os.path.exists(path):
			os.remove(path)
			print('OSError', end ='\r')
	return

class csv2pred:
	##################################################
	# initialization
	##################################################
	def __init__(self,srclist,dst,model_path):
	self.srclist = srclist
	self.dst = dst
	self.model_path = model_path

	if not os.path.exists(self.dst): os.makedirs(self.dst)
	if not os.path.exists(self.dst + 'csvs/'): os.makedirs(self.dst + 'csvs/')

	##################################################
	# step 0: preview csvs
	##################################################
	def precheck(self,path):
	# check broken images
	Pool(32).map(checkfile,path)

	##################################################
	# step 1: setModel
	##################################################
	def setModel(self):
	# load model
	model = torch.load(self.model_path+'model.pth')
	print(model)
	#self.model = model.module
	self.model = model
	self.transforms = transforms.Compose([transforms.Resize(300),
						  transforms.CenterCrop(224),
						  transforms.ToTensor(),
						  transforms.Normalize(RGBmean, RGBstdv)])

	self.indx2class = torch.load(self.model_path+'indx2class.pth')
	print(self.indx2class)

	##################################################
	# step 2: predcsvs(switch mode here)
	##################################################
	def predcsvs(self):
	for src in self.srclist:
		flist = [file for file in glob.glob(src +'/*/*.jpg')]
		if len(flist)==0:
		flist = [file for file in glob.glob(src +'*.jpg')]

		############ only need for the first time
		#self.precheck(flist)

		print('-'*80+'\nProcessing src: {}'.format(src))
		print(len(flist))
		############ comment this line when check result
		self.predImg(flist,src)

	def predImg(self,flist,src):
    	b_size = 150 #batch size
    	w_size = 25
    	img_dir = {i:flist[i] for i in range(len(flist))}
    	img_pre = {i:'None' for i in range(len(flist))}
    	img_vec = {i:'None' for i in range(len(flist))}

    	dsets = ImageReader(img_dir, self.transforms)
    	S_sampler = SequentialSampler(dsets)
    	dataLoader = torch.utils.data.DataLoader(dsets, batch_size=b_size, sampler=S_sampler, num_workers=w_size, drop_last = False)
    	print(len(dsets))

    	#predict images in batches
    	for data in dataLoader:
    		# get the inputs
    		img_bt, idx_bt = data

    #vol =true
    	#    print(type(img_bt))
    		#output = self.model(Tensor((img_bt.cuda(2))
    		output = self.model(img_bt.cuda(2))
    		#output = self.model(Variable(img_bt.cuda(2) ))
    		_, pre_bt = torch.max(output.data, 1)
    		pre_bt = pre_bt.data.cpu().view(-1)
    		output = output.detach().cpu()

    		count = [0,0,0,0]
    		for idx, pre, i in zip(idx_bt,pre_bt,range(len(pre_bt))):
    		#print(pre.size())
    		#print('{}'.format(self.indx2class[pre].item()))
    		#print('{}:{}'.format(idx,self.indx2class[pre].item()),end='\r')
    		count[pre] += 1
    		img_pre[idx] = self.indx2class[pre.item()]
    		img_vec[idx] = np.array2string(output[i,:].view(-1).numpy(), precision=2, separator=',', suppress_small=True)

    	d = {'path': pd.Series(img_dir),
    		 'pred': pd.Series(img_pre),
    		 'pvec': pd.Series(img_vec)}
    	# added
    	if not os.path.isdir(self.dst+'csvs/'):
    		os.mkdir(self.dst+'csvs/')

    	if not os.path.isdir(self.dst+'csvs/tests'):
    		os.mkdir(self.dst+'csvs/tests/')


    	s = src[:-1]

    	print(src[:-1][-1:-4])
    	data=''
    	header=''

    	#df = pd.DataFrame(d).to_csv(self.dst+'csvs/'+src[-4:-1]+'pred.csv')
    	if not os.path.isfile(self.dst+'csvs/tests/'+ src[-4:-1] + '_pred.csv'):
    		with open ( self.dst +'csvs/tests/'+src[-4:-1]+'_pred.csv', 'a') as f:
    			writer = csv.DictWriter(f, delimiter=',',fieldnames=header)
    				writer.writerow(data)

    	df = pd.DataFrame(d).to_csv( self.dst +'csvs/tests/'+src[-4:-1]+'_pred.csv')

    	#df = pd.DataFrame(d).to_csv('result.s3_2/tests/'+src[-4:-1]+'_pred.csv')
    	print(count)
    	a = 0
    	for i in count:
    		a+=i
    	print(a)

	def run(self):
    	print('-'*80+'\nSetting model')
    	self.setModel()
    	print('-'*80+'\nPredicting the result')
    	self.predcsvs()

	def test_scratch(self):
    	srclist = []
    	for d in os.listdir("../../tests/"):

    		srclist.append("../../tests/" + d + "/")
    	dst = 's3.result/scratch'
    	model_path = 'result.s3_2/scratch/'
    	csv2pred(srclist,dst,model_path).run()



if __name__ == '__main__':
    array = []
    i = 0

    try:
        #make sure insert is reading array concatination correctly
        # Create a cursor object
        cursorObject = connectionObject.cursor()
        # cursorObject = connectionObject.cursor()

        # # insertStatement = ",".join("INSERT INTO Images (img_name) VALUES (",str(x),")")
        fields = "(img_name)" #, is_proED)"
        s = "SELECT "
        where = " WHERE "
        cond = " test_category=\"stress\" "
        # selectStatement = s + fields + " FROM " + table + where + cond
        # selectStatement = "SELECT img_name FROM images "
        selectStatement = "SELECT img_name FROM images;"
        # fl.write(selectStatement + ";\n");


        print(selectStatement)

        cursorObject.execute(selectStatement)
        r = cursorObject.fetchall()
        src = []
        for row in r:
            # rr= row['img_name']
            # rrr= row['img_name']
            rr = str(row).replace('---', '\\')
            src.add(rr)
            print(rr)

    except Exception as e:
        print("Exeception occured:{}".format(e))
    finally:
        connectionObject.close()

    print("finishing querying database for test images")
    print("reading images in evaluating them with classifier")

    srclist = []
	# for d in os.listdir("../../stress_tests/"):
	# 	print(d)
	# 	if d[:2] != '._' :
	# 		srclist.append("../../stress_tests/" + d + "/")
	# print(srclist)
	dst = 'result.s3.DB/'
	model_path = 'result.s3_2/ep_32/'
	csv2pred(src,dst,model_path).run()

#source for printing path https://stackoverflow.com/questions/14676407/list-all-files-in-the-folder-and-also-sub-folders
#source for pymysql: https://pythontic.com/database/mysql/insert%20rows?fbclid=IwAR15PJq1UlC3zz1vARvCs2F_Lgpi7qcHbGlEor8XIy2YFkNWnEP1Ktb234U
