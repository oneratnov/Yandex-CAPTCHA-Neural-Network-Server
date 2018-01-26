#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer
#############
import os
import hashlib
#############
from PIL import Image
import cv2
import numpy as np
#from sklearn.svm import SVC
#from sklearn import datasets
#from sklearn.externals import joblib
import pickle
import time
import base64
#from multiprocessing.dummy import Pool as ThreadPool
#############
import requests
import psutil
import Levenshtein
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
################################################################
users = ['qweqwe','shiny','leo','rkbtyn2','mega94636']
def isUser(user):
    global users
    for u in users:
        if u==user:
            return True
    return False
def getBroadcast():
    global users
    us = {}
    for u in users:
        us[u]={}
    return us
def getWords(wordsFile,lenght):
    f = open(wordsFile,'r')
    ws = []
    for w in f.readlines():
        if len(w[0:-2])==lenght*2:
            ws.append(w[0:-2])
    f.close()
    return ws
################################################################
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
class SVC(StatModel):
    def __init__(self):
        self.model = cv2.SVM()
    def train(self, samples, responses):
        params = dict(kernel_type = cv2.SVM_POLY,svm_type = cv2.SVM_C_SVC,degree=3,C=1)
        self.model.train(samples, responses, params = params)
    def predict(self, sample):
        return int(self.model.predict(np.float32(sample)))
class Neuro(object):
    def __init__(self,port,ip='127.0.0.1'):
        self.ip=ip
        self.port=str(port)
    def predict(self,matrix):
        r = requests.post('http://'+self.ip+':'+self.port,data={'matrix':pickle.dumps(matrix)},timeout=5)
        return int(r.content)
class Monitor(object):
    broadcast = getBroadcast()
    nodes ={}
    aging=30
    host='api.audit-seo.ru'
    word=''
    dist=2
    file='monitor'
    myip='127.0.0.1'
    def __new__(cls):
        if not hasattr(cls, 'instance'):
             cls.instance = super(Monitor, cls).__new__(cls)
        return cls.instance
    def getWord(self,user,id):
        return self.broadcast[user][id]['word'][0]
    def setWord(self,user,id,dist,word):
        if id not in self.broadcast[user]:
            self.broadcast[user][id]={'word':[self.word,self.dist]}
        if dist<=self.broadcast[user][id]['word'][1]:
            self.broadcast[user][id]['word']=[word,dist]
    def delId(self,user,id):
        del self.broadcast[user][id]
    def setNode(self,cpu,ip):
        self.nodes[ip]=[cpu,self.__time()]
    def getNode(self):
        current=self.myip
        for ip in self.nodes:
            if self.__actual(self.nodes[ip][1]) and self.__round(current)>self.__round(ip):
                current=ip
        return current
    def __round(self,ip):
        return int(float(self.nodes[ip][0]))
    def __actual(self,time):
        return time<(self.__time()+self.aging)
    def __time(self):
        return int(time.time())
    def __send(self,host,patch,data):
        #print(data)
        os.system('curl --data "'+data+'" -X POST http://'+host+'/'+patch)
        #return requests.post('http://'+host+'/'+patch,data=data,timeout=(1,50))
    def __wl(self,head,first='',second=''):
        return head+'\t'+first+'\t'+second+'\r\n'
    def sayMain(self):
        return self.__send(self.host,'monitor','cpu='+str(psutil.cpu_percent(interval=1)))
    def sayNode(self,key,id):
        return self.__send(self.getNode(),'in.php','key='+key+'&id='+id)
    def save(self):
        monitor=self.__wl('Servers:')
        for ip in self.nodes:
            if self.__actual(self.nodes[ip][1]):
                monitor+=self.__wl('',ip,self.nodes[ip][0])
            else:
                monitor+=self.__wl('',ip,'off')
        monitor+=self.__wl('Clients:')
        for user in self.broadcast:
            monitor+=self.__wl('',user,str(len(self.broadcast[user])))
        f = open(self.file,'w')
        f.write(monitor)
        f.close()
class Captcha(object):
    def __init__(self,r):
        self.user = r.get_argument('key')
        self.id = r.get_argument('id','')
        self.method = r.get_argument('method','')
        self.reqBody = r.get_argument('body','')
        self.files = r.request.files
        self.patch = './data/'+self.user
        self.ansPatch = './complete/'+self.user
        self.filename = self.patch+'/'+self.id+'.gif'
    def check(self):
        return ('file' in self.files and self.method == 'post') or self.method == 'base64'
    def creatDir(self):
        if not os.path.exists(self.patch):
            os.mkdir(self.patch)
        if not os.path.exists(self.ansPatch):
            os.mkdir(self.ansPatch)
    def save(self):
        if self.method == 'base64':
            self.body = base64.b64decode(self.reqBody)
        else:
            file = self.files['file'][0]#body
            self.body = file['body']
        self.id = hashlib.md5(self.body+time.strftime('%d %b %Y %H:%M:%S')).hexdigest()
        self.filename = self.patch+'/'+self.id+'.gif'
        f = open(self.filename, 'w')
        f.write(self.body)
        f.close()
    def complete(self):
        apatch = self.ansPatch+'/'+self.id
        if not os.path.exists(apatch):
            os.mkdir(apatch)
        os.system('cp '+self.filename+' '+apatch+'/'+self.answer.decode('utf-8')+'.gif')
        Monitor().delId(self.user,self.id)
################################################################
class yandexCaptcha(object):
    neuro=Neuro(port=8009)
    neuroLen=Neuro(port=8008)
    words4=getWords('words.txt',4)
    words5=getWords('words.txt',5)
    words6=getWords('words.txt',6)
    words7=getWords('words.txt',7)
    height = 40
    matrixW = 40
    matrixLenW = 200
    def getImgBin(self,img,clr):
      im = img.convert('RGB')
      im2 = Image.new('L',im.size,'white')
      for y in range(im.size[1]):
        for x in range(im.size[0]):
          r,g,b = im.getpixel((x,y))
          if (r==g and g==b and r<clr) and (x>170 and y<15) != True:
            im2.putpixel((x,y),0)
      return np.array(im2).copy()
    def getMagicData(self,img):
      cvImg = self.getImgBin(img,160)
      thresh = cv2.adaptiveThreshold(cvImg, 255, 1, 1, 11, 2) 
      countours,_ = cv2.findContours(thresh,cv2.RETR_LIST ,cv2.CHAIN_APPROX_TC89_KCOS)
      maxArea = 0
      maxCnt = []
      for cnt in countours:
        currentArea = cv2.contourArea(cnt)
        if maxArea < currentArea:
          maxArea = currentArea
          maxCnt = cnt
      maxW=maxH=0
      minX=minY=200
      for cnt in countours:
        if maxArea == cv2.contourArea(cnt):
          continue
        x,y,w,h = cv2.boundingRect(cnt)
        if minX > x :
          minX = x
        if minY > y :
          minY = y
        if maxW < x+w :
          maxW = x+w
        if maxH < y+h :
          maxH = y+h
      return (maxCnt,minX,minY,maxW-minX,maxH-minY)
    def getSymData(self,sym,width):
      M, N = sym.shape
      i = 1
      c = 0
      matrix = []
      for p in sym.reshape(M*N):
        if i == 1:
          matrix.append([])
        if p==255:
          matrix[c].append(-1)
        else:
          matrix[c].append(1)
        if i<N:
          i+= 1
        else:
          i = 1
          c+= 1
      for row in range(0,M):
        for x in range(0,width-N):
          matrix[row].append(-1)
      m = []
      for r in matrix:
        for p in r:
          m.append(p)
      return np.array(m)
    def __new__(cls):
        if not hasattr(cls, 'instance'):
             cls.instance = super(yandexCaptcha, cls).__new__(cls)
        return cls.instance
    def getNameFor(self,lenght,cvImg):
        h,w = cvImg.shape
        name = ''
        for i in range(1,lenght+1):
          sym = cvImg[0:h, (i-1)*w/lenght:i*w/lenght]
          if (i*w/lenght-(i-1)*w/lenght)>self.matrixW:
            sym = cv2.resize(sym,(self.matrixW, self.height), interpolation = cv2.INTER_CUBIC)
          name += unichr(self.neuro.predict(self.getSymData(sym,self.matrixW)))
        return name
    def hack(self,captcha):
        img = Image.open(captcha.filename)
        maxCnt,minX,minY,W,H = self.getMagicData(img)
        width = W*self.height/H
        cvImg = self.getImgBin(img,220)
        cv2.drawContours(cvImg,[maxCnt],0,(255,0,0),2)
        cvImg = cv2.morphologyEx(cvImg, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
        cvImg = cvImg[minY: minY+H, minX: minX+W]
        cvImg = cv2.resize(cvImg,(width, self.height), interpolation = cv2.INTER_CUBIC)
        h,w = cvImg.shape
        cvImgLen = cv2.resize(cvImg,(self.matrixLenW, self.height), interpolation = cv2.INTER_CUBIC) if w>self.matrixLenW else cvImg
        lenght = self.neuroLen.predict(self.getSymData(cvImgLen,self.matrixLenW))
        answer = self.getNameFor(lenght,cvImg)
        for word in getattr(self,'words'+str(lenght)):
            dist = Levenshtein.hamming(word,answer.encode('utf-8'))
            Monitor().setWord(captcha.user,captcha.id,dist,word)
            if dist==0:
                break
        captcha.answer = Monitor().getWord(captcha.user,captcha.id)
################################################################
class In(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def post(self):
        captcha = Captcha(self)
        if not isUser(captcha.user):
            return self.write('USER_NOT_FOUND')
        if captcha.id=='':
            if captcha.check():
                captcha.creatDir()
            else:
                return self.write('FILE_NOT_FOUND')
            captcha.save()
        self.write('OK|'+captcha.id)
        self.finish()
        if self.request.remote_ip=='127.0.0.1':
            yandexCaptcha().hack(captcha)
            captcha.complete()
        else:
            Monitor().sayNode(captcha.user,captcha.id)
class Out(tornado.web.RequestHandler):
    def get(self):
        key = self.get_argument('key')
        id = self.get_argument('id')
        patch = './complete/'+key+'/'+id
        if not isUser(key):
            return self.write('USER_NOT_FOUND')
        if not os.path.exists(patch):
            return self.write('CAPCHA_NOT_READY')
        capDir = os.listdir(patch)
        if len(capDir)>0:
            answer = capDir[0].encode('utf-8')[:-4]
            if answer=='':
                self.write('ERROR_CAPTCHA_UNSOLVABLE')
            else:    
                self.write('OK|'+answer)
        else:
            self.write('CAPCHA_NOT_READY')
class Mon(tornado.web.RequestHandler):
    def get(self):
        Monitor().sayMain()
        Monitor().save()
    def post(self):
        Monitor().setNode(self.get_argument('cpu'),self.request.remote_ip)
class Reboot(tornado.web.RequestHandler):
    def get(self):
        os.system('reboot')

if __name__ == "__main__":
    app = tornado.web.Application([ 
        (r"/in\.php", In),
        (r"/res\.php", Out),
        (r"/monitor", Mon),
        (r"/07f346bb747b3bcb89d797b84e2605a9", Reboot),
    ])
    server = HTTPServer(app)
    #server.listen(80)
    #tornado.ioloop.IOLoop.current().start()
    server.bind(80)
    server.start(72)
    tornado.ioloop.IOLoop.instance().start()
