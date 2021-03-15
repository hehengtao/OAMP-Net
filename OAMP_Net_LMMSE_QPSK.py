# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:48:29 2018

@author: dell
"""

#!/usr/bin/env python
import time
import tensorflow as tf
import numpy as np
import scipy.io as sc 
import os
from scipy.linalg import toeplitz
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#parameters
reuse=tf.AUTO_REUSE
N = 4
M = 4
snrdb_train =30 # training SNR
snr_train = 10.0 ** (snrdb_train/10.0)  #train_SNR_linear
batch_size=100
epochs=1000
itermax=50

train_size=5000
valid_size=10000
errsum=1000  #计算错误总数
rho=0.8
snrdb_test=snrdb_train
snr_test = 10.0 ** (snrdb_test/10.0)  #train_SNR_linear
weight_mat='C:\GT\GT works\globalsip长文工作\Code_Globalsip_QPSK'

Rx = np.logspace( 0, np.log10( rho**(M-1)),M)

Tx = np.logspace( 0, np.log10( rho**(N-1)),N)

sigma2=1/snr_test

def Network_ini(theta):
    update=[]
    for var in tf.trainable_variables():
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
    return update

def Variable(shape):
    gamma=tf.get_variable('gamma', shape=shape, initializer=tf.ones_initializer)
    return gamma
    
def generate_data_iid_test(B,M,N,SNR):  
    y_real=np.zeros([2*M,B])   
    x_real=np.zeros([2*N,B])
    H_real=np.zeros([B,2*M,2*N])
    
#    snrdb_test_inv=10.**(-snrdb_test/10.)
    
    H_=np.zeros([B,M,N],dtype=complex)
      
#    H_ = (np.random.randn(B,M,N)+1j*np.random.randn(B,M,N))
    H_ = (np.random.randn(B,M,N)+1j*np.random.randn(B,M,N))/np.sqrt(2*M)
#    W_=np.zeros([B,K,K])
    x_=np.zeros([N,B],dtype=complex)
    x_=Modulation((np.random.uniform( 0,1,(N,B))<0.5))+1j*Modulation((np.random.uniform( 0,1,(N,B))<0.5))
   
    y_=np.zeros([M,B],dtype=complex)   
    w=np.sqrt(1/2)*(np.random.randn(M,B)+1j*np.random.randn(M,B))*np.sqrt(sigma2)
    
    for i in range(B):
#      H=Correlated(N,M,Rx,Tx)
#      print(H)
      H=H_[i,:,:]
#      tmp_snr=(H.T.dot(H)).trace()/N
#      H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
#      H_[i,:,:]=H

      y_[:,i]=np.matmul(H,x_[:,i])+w[:,i]
#     print(y_[:,i])
      #y_real[:,i]=np.vstack( (np.expand_dims(np.real(y_[:,i]),-1),  np.expand_dims(np.real(y_[:,i]),-1)     )     ) #stack 要注意加()
      y_real[:,i]=np.hstack((np.real(y_[:,i]), np.imag(y_[:,i]))).T #stack 要注意加()  
      
      H_real[i,:,:]=np.vstack((np.hstack((np.real(H),-np.imag(H))),np.hstack((np.imag(H),np.real(H)))))

      x_real[:,i]=np.hstack((np.real(x_[:,i]),np.imag(x_[:,i]))).T

    return y_real,H_real,x_real
    
def Correlated(N,M,Rx,Tx):  
    
    T=toeplitz(Tx)
    h=np.math.sqrt(1/2)*(np.random.normal(0,1,(M*N,1))+1j*np.random.normal(0,1,(M*N,1)))
    R=toeplitz(Rx)

    C=np.kron(T,R)

    AAA=np.matmul(np.linalg.cholesky(C),h)
 
    A=np.reshape(AAA,(M,N))

    return A    
        
def Modulation(bits):                                        
    x_re= (2*bits-1)/np.sqrt(2)               
    return x_re
        
def shrink_bg_QPSK(r,rvar):
    eps=5e-13
    b=2*phi(-1/np.sqrt(2),r,rvar)
    tp=tf.maximum(phi(-1/np.sqrt(2),r,rvar),eps)+tf.maximum(phi(1/np.sqrt(2),r,rvar),eps)
    a=1-tf.div(b,(tp))
#    print(b,tp,a)
    xhat_=(a)/np.sqrt(2)
    return (xhat_)    
    
def phi(x,r,rvar): 
#    print(r,'MMM')      
    rvar=tf.tile(tf.expand_dims(tf.expand_dims(rvar, axis=-1), axis=-1), [1,2*N,1])
#    print(r)
#    print(rvar)
    y=tf.exp(-tf.square(x-r)/(2*rvar))
#    print(rvar,'yyy')
    return y  
def Train_batch(sess):
    _loss = list()
    _ser = list()
    packet=valid_size//batch_size
    for offset in range(packet):
        batch_Y, batch_H, batch_X = generate_data_iid_test(batch_size,M,N,snr_train)
        _, b_loss, b_ser = sess.run([optimizer,cost,ser], {Y_: batch_Y.T, A_: batch_H, X_: batch_X.T})
        _loss.append(b_loss)
        _ser.append(b_ser)
    return np.mean(_loss), np.mean(_ser)
    
def Valid_batch(sess):
    _loss = []
    _ser = []
    packet=valid_size//batch_size
    for offset in range(packet):
        batch_Y, batch_H, batch_X = generate_data_iid_test(batch_size,M,N,snr_train)
        b_loss, b_ser = sess.run([cost,ser], {Y_: batch_Y.T, A_: batch_H, X_: batch_X.T})
        _loss.append(b_loss)
        _ser.append(b_ser)
    return np.mean(_loss), np.mean(_ser)
    
def Train():
    print("\nTraining ...")     
    with tf.Session() as sess:        
        tf.global_variables_initializer().run()              
        weight_file=weight_mat+'oamp.mat'
        best_valid_loss=0    
        for i in range(epochs): 
            start_time = time.time()          
            train_loss, ser_train = Train_batch(sess)
            valid_loss, ser_valid = Valid_batch(sess)
            time_taken = time.time() - start_time
            print("Epoch %d Valid Loss: %.6f, Valid SER: %.6f, Time Cost: %.2f s"
                  % (i+1,valid_loss, ser_valid, time_taken))
            if i==0 or (i>0 and valid_loss < best_valid_loss):
                best_valid_loss = valid_loss #保存最优的网络loss
        #        print(best_valid_loss)
                Save(weight_file)
        print("\nTraining is finished.") 
        
def Test_batch(sess, ebn0_test):
    _loss = []
    _ser = []
    sernum = 0
    while sernum<errsum:
        batch_Y, batch_H, batch_X= generate_data_iid_test(batch_size,M,N,snr_train)
        b_loss, b_ser, errnum, s_, gamma_  = sess.run([cost,ser,err,s, gamma], {Y_: batch_Y.T, A_: batch_H, X_: batch_X.T})
        _loss.append(b_loss)
        _ser.append(b_ser)      
        sernum=sernum+errnum
    return np.mean(_loss), np.mean(_ser), s_, gamma_,tr

def Test(snr_test):
    with tf.Session() as sess: 
        tf.global_variables_initializer().run()
        sess.run(update)
        loss_test, ser_test,s_, gamma_,tr_ = Test_batch(sess, snr_test)
        print("Eb/N0: %.0f, Testing Loss: %.6f, Testing SER: %.6f"% (snr_test, loss_test, ser_test))
    return loss_test, ser_test,s_, gamma_,tr_  

def Save(weight_file):
    dict_name={}
    for varable in tf.trainable_variables():  
        dict_name[varable.name]=varable.eval()
    sc.savemat(weight_file, dict_name)     

with tf.Graph().as_default():
    #tensorflow placeholders, the input given to the model in order to train and test the network
    A_ = tf.placeholder(tf.float32,shape=[None,2*M,2*N])
    X_ = tf.placeholder(tf.float32,shape=[None,2*N])
    Y_ = tf.placeholder(tf.float32,shape=[None,2*M])
    
    s=tf.zeros((batch_size,2*N,1),dtype=tf.float32)#给s多加一个维度
    damping=0.1
    tau2=1
    sigma2=10.**(-snrdb_test/10.)
    eps=1e-9
    v2=tf.ones((batch_size,))
    beta=5e-1
    
    I=tf.eye(2*M,batch_shape=[batch_size])
  
 #   IM=tf.eye(2*M,batch_shape=[batch_size])    
 #   IN=tf.eye(2*N,batch_shape=[batch_size])  
      
    with tf.variable_scope('gamma', reuse=reuse):
        gamma=Variable((itermax,))
#    W=tf.matmul(A_,tf.matrix_inverse(tf.matmul(A_,A_,adjoint_b = True)+beta*I),adjoint_a = True)#LMMSE Matrix     
    for t in range(itermax):
        v2=tf.tile(tf.expand_dims(tf.expand_dims(v2, axis=-1), axis=-1),[1,2*M,2*N])
             
        RR=tf.matrix_inverse(tf.multiply(v2, tf.matmul(A_,A_,adjoint_b = True))+sigma2*I/2)
#        print(RR)
        W=tf.multiply(v2, tf.matmul(A_,RR,adjoint_a = True))  #LMMSE Matrix
#        W=tf.matmul(v2*A_,tf.matrix_inverse(tf.matmul(v2*A_,A_,adjoint_b = True)+sigma2*tf.eye(N)),adjoint_a = True)  #LMMSE Matrix
        tr=tf.trace(tf.matmul(W,A_))
  #      print(tr)
        tr=tf.tile(tf.expand_dims(tf.expand_dims(tr, axis=-1), axis=-1),[1,2*M,2*N])
        W=2*N/tr*W 
        
        z = tf.expand_dims(Y_,-1)-tf.matmul(A_,s)
        r = s + gamma[t]*tf.matmul(W,z)
        v2 = tf.maximum(tf.div(tf.norm(z, axis=[-2,-1])**2-M*sigma2,tf.trace(tf.matmul(A_,A_,adjoint_a=True))), eps)
#        tau2 = v2/(2*N)*(2*N+(gamma[t]**2-2*gamma[t])*2*M)+gamma[t]**2*sigma2/(2*N)*tf.trace(tf.matmul(W,W,adjoint_b=True))
#        tau2=M/(M-N)*sigma2*tf.ones((batch_size,))
        B=I-gamma[t]*tf.matmul(W, A_)
        tau2 = v2/2/N*tf.trace(tf.matmul(B,B,adjoint_b=True))+gamma[t]*gamma[t]*sigma2/4/N*tf.trace(tf.matmul(W,W,adjoint_b=True))
        s = shrink_bg_QPSK(r ,tau2)
    
    s=s[:,:,0]
    cost  = tf.nn.l2_loss(s-X_)  # l2 loss function
    err_temp = tf.to_float(tf.not_equal(tf.sign(s),tf.sign(X_)))
    err=tf.reduce_sum(err_temp)
    ser = tf.reduce_mean(err_temp)
      
    
    #learning_rate=0.001
    with tf.variable_scope('opt', reuse=reuse):
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
    
    #Training DetNet
    Train()
    update=Network_ini(sc.loadmat(weight_mat+'oamp.mat'))
    loss_test, ser_test, s_, gamma_,tr_ =Test(snr_train)
    
          
