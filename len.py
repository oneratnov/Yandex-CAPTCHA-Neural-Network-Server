#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer
import pickle
import cv2
import numpy as np
import requests
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
        params = dict(kernel_type = cv2.SVM_POLY,svm_type = cv2.SVM_C_SVC,degree=5,C=1)
        self.model.train(samples, responses, params = params)
    def predict(self, sample):
        return int(self.model.predict(np.float32(sample)))
################################################################
class Neuro(tornado.web.RequestHandler):
    neuro=SVC()
    neuro.load('len.xml')
    def post(self):
        if self.request.remote_ip=='127.0.0.1':
            matrix = self.get_argument('matrix')
            answer = self.neuro.predict(pickle.loads(matrix))
            self.write(str(answer))
if __name__ == "__main__":
    app = tornado.web.Application([ 
        (r"/", Neuro),
    ])
    server = HTTPServer(app)
    server.bind(8008)
    server.start(0)
    tornado.ioloop.IOLoop.instance().start()
