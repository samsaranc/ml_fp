from __future__ import print_function
import os, copy
import glob
import PNet
from multiprocessing import Pool

import wget 
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

LATEST_MODEL_PATH = 'models/s3_77/'
LATEST_STATE_DICT = '77_state_dict.pth'
PHASE = ['train','val']
AWNS = ['AWNED','AWNLESS ']
# normalization paras
RGBmean, RGBstdv = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def checkfile(path):
        try:
                img = default_loader(path)
        except OSError:
                if os.path.exists(path):
                        os.remove(path)
                        print('OSError', end ='\r')
        return

def test_url(filename):
        model_path = LATEST_MODEL_PATH
        sd_path = model_path + LATEST_STATE_DICT
        dst = '.'
        flist = []
        flist.append(filename)
        return csv2pred(flist,dst,model_path, sd_path).run_url()


class csv2pred:
        ##################################################
        # initialization
        ##################################################
        def __init__(self,srclist,dst,model_path,sd_path):
                self.srclist = srclist
                self.dst = dst
                self.model_path = model_path
                self.state_dict_path = sd_path
                self.indx2class = torch.load(self.model_path+'indx2class.pth')
                if not os.path.exists(self.dst): os.makedirs(self.dst)
                # if not os.path.exists(self.dst + 'csvs/'): os.makedirs(self.dst + 'csvs/')

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
                #print(model)
                #self.model = model.module
                self.model = model
                self.transforms = transforms.Compose([transforms.Resize(300),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(RGBmean, RGBstdv)])

                self.indx2class = torch.load(self.model_path+'indx2class.pth')
                #print(self.indx2class)

        ##################################################
        # step 1: setModel
        ##################################################
        def setModel_NEW(self):
        # load model
                num_classes = 2
                device = torch.device("cuda")
                model = PNet.resnetP(num_classes)
                sd = torch.load(self.state_dict_path)
                print("loaded sd")
                model.load_state_dict(sd)
                model.to(device)
                model.eval()
                #print(model)
                #self.model = model.module
                self.model = model
                self.transforms = transforms.Compose([transforms.Resize(300),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(RGBmean, RGBstdv)])

                # self.indx2class = torch.load(self.model_path+'indx2class.pth')
                #print(self.indx2class)

        ##################################################
        # step 2: predcsvs(switch mode here)
        ##################################################
        def predcsvs(self):
                for src in self.srclist:
                        flist = [file for file in glob.glob(src +'/*/*.jpg')]
                        if len(flist)==0: flist = [file for file in glob.glob(src +'*.jpg')]

                        ############ only need for the first time
                        #self.precheck(flist)

                        print('-'*80+'\nProcessing src: {}'.format(src))
                        print(len(flist))
                        ############ comment this line when check result
                        self.predImg(flist,src)

#       def predurl(self):
#               flist = self.srclist
#               self.predImg_int(flist)

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
        
        def predImg_int(self,flist):
                b_size = 150 #batch size
                w_size = 25
                img_dir = {i:flist[i] for i in range(len(flist))}
                img_pre = {i:'None' for i in range(len(flist))}
                img_vec = {i:'None' for i in range(len(flist))}

                dsets = ImageReader(img_dir, self.transforms)
                S_sampler = SequentialSampler(dsets)
                dataLoader = torch.utils.data.DataLoader(dsets, batch_size=b_size, sampler=S_sampler, num_workers=w_size, drop_last = False)
                #print(len(dsets))

                #predict images in batches
                for data in dataLoader:
                        # get the inputs
                        img_bt, idx_bt = data
                        #device = torch.device("cuda")
                        #output = self.model(Tensor((img_bt.cuda(2))
                        output = self.model(img_bt.cuda(0))
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
                                pre = img_pre[idx]
                                vec = img_vec[idx]
                                #print(str(idx) + ' '  + img_vec[idx])
                                #print(img_pre[idx])
                #print(flist)
                print('pred '+str(pre))
                #print('vec '+str(vec))
                
                if pre=='NPA': 
                        pre =0
                elif pre=='PA':
                        pre =1
                else:
                        pre=2
                        
                return pre

        def run_url(self):
                print('-'*80+'\nSetting model')
                self.setModel_NEW()
                print('-'*80+'\nPredicting the result')
                pred = self.predurl()
                #print(pred)
                return pred

        def run(self):
                print('-'*80+'\nSetting model')
                self.setModel()
                print('-'*80+'\nPredicting the result')
                self.predcsvs()


        def predurl(self):
                flist = self.srclist
                return self.predImg_int(flist)

        def test_scratch(self):
                srclist = []
                for d in os.listdir("../../tests/"):

                        srclist.append("../../tests/" + d + "/")
                dst = 's3.result/scratch'
                model_path = 'result.s3_2/scratch/'
                csv2pred(srclist,dst,model_path).run()

if  __name__ == '__main__':
        '''
        srclist = []
        for d in os.listdir("../../stress_tests/"):
                print(d)
                if d[:2] != '._' :
                        srclist.append("../../stress_tests/" + d + "/")
        print(srclist)
        csv2pred(srclist,dst,model_path).run()'''
        url = 'https://images-na.ssl-images-amazon.com/images/I/514Lbvt36zL._SX377_BO1,204,203,200_.jpg'
        print('configured for urls')
        #url = 'https://samsaranc.github.io/images/avatar.jpg'
        try: 
                filename = wget.download(url)
                test_url(filename)
        except Exception as e:
                print(e) 
