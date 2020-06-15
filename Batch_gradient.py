# -*- coding: utf-8 -*-
"""
Using different batch sizes to test gradient descent method in linear regression

@author: mayao
"""

import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import time

def get_data_ch7():  
    data = np.genfromtxt('C:/Users/mayao/d2l-zh/data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])

def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # Iniatial model
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    #Store the loss
    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    # Create Trainer
    trainer = gluon.Trainer(
        net.collect_params(), trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)  # Average the gradient
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # Print result and graph
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')

# One experiment
features, labels = get_data_ch7()
#train_gluon_ch7('sgd', {'learning_rate': 0.05}, features, labels, 10)
#train_gluon_ch7('sgd', {'learning_rate': 1}, features, labels, 1500)
train_gluon_ch7('sgd', {'learning_rate': 0.001}, features, labels, 1)