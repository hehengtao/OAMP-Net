# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:26:29 2019

@author: hhengtao3
"""
import time
import tensorflow as tf
import numpy as np
import scipy.io as sc
import os
from scipy.linalg import toeplitz

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from EP_V_Net_CE_Module import EP_V_Channel_estimation
#np.random.seed(0)
#tf.set_random_seed(0)
# reuse=tf.AUTO_REUSE
N = 4  # Number of transmitter antennas
M = 4  # Number of receiver antennas
Tp = 4  # the length of pilot
Tc = 4 * M  # the length of  coherence time
Td = Tc - Tp  # the length of data
snrdb_train = np.array([25], dtype=np.float64)  # training SNR
snr_train = 10.0 ** (snrdb_train / 10.0)  # train_SNR_linear
Sigma2 = 1 / snr_train
epochs = 1000  # Number of epoches
train_size = 5000  # Number of training size
valid_size = 10000  # Number of valid size
errsum = 10000  # 计算错误总数
snrdb_test = snrdb_train
snr_test = 10.0 ** (snrdb_test / 10.0)  # train_SNR_linear
weight_mat = 'D:/research results/Model_Driven_DL_JCD/Model_Driven_DL_JCD/Code/Code_TSP_Paper/JCESD/OAMP_Turbo_JCD_Python'

batch_size = 100  # bathsize
itermax = 6  # Numbero of EP iterations
iter_JCD = 4  # number of JCD iterations
mapping_table_QPSK = {
    (0, 0): 1 / np.sqrt(2) + (1 / np.sqrt(2)) * 1j,
    (0, 1): 1 / np.sqrt(2) - (1 / np.sqrt(2)) * 1j,
    (1, 0): -(1 / np.sqrt(2)) + (1 / np.sqrt(2)) * 1j,
    (1, 1): -(1 / np.sqrt(2)) - (1 / np.sqrt(2)) * 1j,
}
demapping_table_QPSK = {v: k for k, v in mapping_table_QPSK.items()}


mapping_table_16QAM = {
    (0, 0, 0, 0): 1 / np.sqrt(10) + (1 / np.sqrt(10)) * 1j,
    (0, 0, 0, 1): 1 / np.sqrt(10) + (3 / np.sqrt(10)) * 1j,
    (0, 0, 1, 0): 3 / np.sqrt(10) + (1 / np.sqrt(10)) * 1j,
    (0, 0, 1, 1): 3 / np.sqrt(10) + (3 / np.sqrt(10)) * 1j,
    (0, 1, 0, 0): 1 / np.sqrt(10) + (-1 / np.sqrt(10)) * 1j,
    (0, 1, 0, 1): 1 / np.sqrt(10) + (-3 / np.sqrt(10)) * 1j,
    (0, 1, 1, 0): 3 / np.sqrt(10) + (-1 / np.sqrt(10)) * 1j,
    (0, 1, 1, 1): 3 / np.sqrt(10) + (-3 / np.sqrt(10)) * 1j,
    (1, 0, 0, 0): -1 / np.sqrt(10) + (1 / np.sqrt(10)) * 1j,
    (1, 0, 0, 1): -1 / np.sqrt(10) + (3 / np.sqrt(10)) * 1j,
    (1, 0, 1, 0): -3 / np.sqrt(10) + (1 / np.sqrt(10)) * 1j,
    (1, 0, 1, 1): -3 / np.sqrt(10) + (3 / np.sqrt(10)) * 1j,
    (1, 1, 0, 0): -1 / np.sqrt(10) + (-1 / np.sqrt(10)) * 1j,
    (1, 1, 0, 1): -1 / np.sqrt(10) + (-3 / np.sqrt(10)) * 1j,
    (1, 1, 1, 0): -3 / np.sqrt(10) + (-1 / np.sqrt(10)) * 1j,
    (1, 1, 1, 1): -3 / np.sqrt(10) + (-3 / np.sqrt(10)) * 1j,
}
demapping_table_16QAM = {v: k for k, v in mapping_table_16QAM.items()}


mu = 2
mode = 'QPSK'

S_QPSK = [1 / np.sqrt(2), -1 / np.sqrt(2)]

D = np.zeros([mu, ], dtype='float64')

index = 0

def Signomid_fun(x):
    y = 1 / (1 + np.exp(-x))
    return y


def Network_ini(theta):
    update = []
    for var in tf.trainable_variables():
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),
                                tf.constant(np.reshape(theta[var.name], var.shape))))
    return update


def Variable(shape):
    gamma=tf.get_variable('gamma', shape=shape, initializer=tf.ones_initializer, dtype='float64')
    return gamma
    
def Variable1(shape):    
    gamma1=tf.get_variable('gamma1', shape=shape, initializer=tf.ones_initializer, dtype='float64')
    return gamma1   

def Variable2(shape):    
    corre=tf.get_variable('corre', shape=shape, initializer=tf.ones_initializer, dtype='float64')
    return corre  

def Variable3(shape):    
    expec=tf.get_variable('expec', shape=shape, initializer=tf.zeros_initializer, dtype='float64')
    return expec


def Modulation(bits):
    symbol = np.zeros([int(len(bits) / mu)], dtype=complex)
    # print(symbol.shape)
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    # print(bit_r.shape)
    if mode == 'QPSK':  # mu=2
        for m in range(0, int(len(bits) / mu)):
            symbol[m] = mapping_table_QPSK[tuple(bit_r[m, :])]
        return symbol
        # return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)    # This is just for QAM modulation
    elif mode == '16QAM':  # mu=4
        for m in range(0, int(len(bits) / mu)):
            symbol[m] = mapping_table_16QAM[tuple(bit_r[m, :])]
        return symbol
    else:
        print("Not a valid modulation mode,", mode)


def MMSE_Constellation(r,rvar):


    b = 2 * phi(-1 / np.sqrt(2), r, rvar)
    #    tp = phi(-1/np.sqrt(2),r,rvar) + phi(1/np.sqrt(2),r,rvar) + eps
    tp = phi(-1 / np.sqrt(2), r, rvar) + phi(1 / np.sqrt(2), r, rvar)

    a = 1 - tf.div(b, tp)
    #    print(b,tp,a)
    xhat_ = a / np.sqrt(2)

    return (xhat_)

def MMSE_AWGN_Gaussian(r, rvar, xhat_init, xvar_init):

    xhat_den = tf.multiply(xvar_init, r) + tf.multiply(rvar, xhat_init)

    xhat_num = rvar + xvar_init

    xhat = tf.divide(xhat_den, xhat_num)

    var_den = tf.multiply(xvar_init, rvar)

    var = tf.divide(var_den, xhat_num)

    return xhat, var


def phi(x, r, rvar):
    y = tf.exp(tf.divide(-tf.square(x - r), 2 * rvar))
    return y

def generate_data_iid_test(B, M, N, SNR_dB):
    #    y_ = np.zeros([M,B],dtype=complex)
    H_ = np.zeros([B, M, N], dtype=complex)
    HH_real = np.zeros([B, 2 * M, 2 * N])
    H_vec = np.zeros([B, M * N, 1], dtype=complex)
    HH_vec = np.zeros([B, 2 * M * N, 1])
    #    Y_pilot = np.zeros([B,M,Tp],dtype=complex)
    #    Y_data = np.zeros([B,M,Td],dtype=complex)
    Y_pilot = np.zeros([M, Tp], dtype=complex)
    Y_data = np.zeros([M, Td], dtype=complex)
    Y_pilot_JCD = np.zeros([M, Tc], dtype=complex)

    X_pilot = np.zeros([N, Tp], dtype=complex)
    X_pilot_batch = np.zeros([B, N, Tp], dtype=complex)

    X_data = np.zeros([N, Td], dtype=complex)
    #    X_pilot_batch = np.zeros([B,N,Tp],dtype=complex)
    Y_JCD_vec = np.zeros([B, Tc * M, 1], dtype=complex)
    Y_JCD_vec_real = np.zeros([B, 2 * Tc * M, 1])

    X_data_batch = np.zeros([B, N, Td], dtype=complex)
    Y_pilot_vec = np.zeros([B, Tp * M, 1], dtype=complex)
    Xpilot_kron = np.zeros([B, Tp * M, M * N], dtype=complex)
    Ypilot_vec_real = np.zeros([B, 2 * Tp * M, 1], dtype='float64')
    Xpilot_kron_real = np.zeros([B, 2 * Tp * M, 2 * M * N], dtype='float64')
    Xdata_batch_real = np.zeros([B, 2 * N, Td], dtype='float64')
    Ydata_batch_real = np.zeros([B, 2 * M, Td], dtype='float64')
    X_data_real = np.zeros([2 * N, Td], dtype='float64')
    Y_data_real = np.zeros([2 * M, Td], dtype='float64')
    snr = 10.0 ** (SNR_dB / 10.0)  # train_SNR_linear
    sigma2 = 1 / snr
    H_ = (np.random.randn(B, M, N) + 1j * np.random.randn(B, M, N)) / np.sqrt(2 * N)
    sigma2_pilot = 1 / snr  # train_SNR_linear

    #    dftmtx = lambda N: np.fft.fft(np.eye(N))
    for i in range(B):
        #      H=Correlated(N,M,Rx,Tx)/np.sqrt(2*M)
        H = H_[i, :, :]
        HH_real[i, :, :] = np.vstack((np.hstack((np.real(H), -np.imag(H))), np.hstack((np.imag(H), np.real(H)))))
        W_pilot = (np.random.randn(M, Tp) + 1j * np.random.randn(M, Tp)) * np.sqrt(sigma2_pilot / 2)
        W_data = (np.random.randn(M, Td) + 1j * np.random.randn(M, Td)) * np.sqrt(sigma2 / 2)

        for jj in range(Tp):
            bit_pilot = (np.random.uniform(0, 1, (mu * N,)) < 0.5)
            X_pilot[:, jj] = Modulation(bit_pilot)

        X_pilot_batch[i, :, :] = X_pilot
        Y_pilot = np.matmul(H, X_pilot) + W_pilot
        Y_pilot_vec[i, :, :] = np.reshape(Y_pilot.T, (Tp * M, 1))
        Xpilot_kron[i, :, :] = np.kron(X_pilot.T, np.eye(M))

        for kk in range(Td):
            bit_data = (np.random.uniform(0, 1, (mu * N,)) < 0.5)
            X_data[:, kk] = Modulation(bit_data)
            X_data_real[:, kk] = np.hstack((np.real(X_data[:, kk]), np.imag(X_data[:, kk]))).T

        Y_data = np.matmul(H, X_data) + W_data

        Y_data_real = np.vstack((np.real(Y_data), np.imag(Y_data)))

        Y_pilot_JCD = np.hstack((Y_pilot, Y_data))

        Y_JCD_vec[i, :, :] = np.reshape(Y_pilot_JCD.T, (Tc * M, 1))

        X_data_batch[i, :, :] = X_data
        Ypilot_vec_real[i, :, :] = np.vstack(
            (np.real(Y_pilot_vec[i, :, :]), np.imag(Y_pilot_vec[i, :, :])))  # stack 要注意加()
        Y_JCD_vec_real[i, :, :] = np.vstack(
            (np.real(Y_JCD_vec[i, :, :]), np.imag(Y_JCD_vec[i, :, :])))  # stack 要注意加()

        Xpilot_kron_real[i, :, :] = np.vstack((np.hstack(
            (np.real(Xpilot_kron[i, :, :]), -np.imag(Xpilot_kron[i, :, :]))), np.hstack(
            (np.imag(Xpilot_kron[i, :, :]), np.real(Xpilot_kron[i, :, :])))))
        Xdata_batch_real[i, :, :] = X_data_real
        Ydata_batch_real[i, :, :] = Y_data_real
        H_vec[i, :, :] = np.reshape(H.T, (N * M, 1))
        HH_vec[i, :, :] = np.vstack((np.real(H_vec[i, :]), np.imag(H_vec[i, :])))
        
    return Ypilot_vec_real, Y_JCD_vec_real, Xpilot_kron, Xpilot_kron_real, np.real(X_pilot_batch), np.imag(
        X_pilot_batch), Ydata_batch_real, Xdata_batch_real, H_, HH_vec, HH_real, X_data_batch


def Train_batch(sess):
    _loss = list()
    SER_train = list()
    packet = valid_size // batch_size
    x_bit_hat = np.zeros([mu, ], dtype=float)
    x_bit_true = np.zeros([mu, ], dtype=float)
    bit_hat = np.zeros([mu * N * batch_size, ], dtype=float)
    bit_true = np.zeros([mu * N * batch_size, ], dtype=float)
    _ber = list()
    for offset in range(packet):
        batch_Ypilot, batch_YJCD, Xpilot_kron, Xpilot_kron_real, batch_Xpilot_real, batch_Xpilot_imag, batch_Ydata, batch_Xdata, HH_sample, HH_vec_, HH_real_, x_complex_ = generate_data_iid_test(
            batch_size, M, N, snrdb_train)
#        print(batch_Ydata.shape)
        _, b_loss, X_data_hat_, Xpilot_real_equal_, HH_est_, MSE_ = sess.run(
            [optimizer, cost, X_Ouput, Xpilot_real_equal, HH_est, MSE],
            {Ypilot_vec_: batch_Ypilot, Ypilot_JCD_: batch_YJCD, Xpilotkron_real: Xpilot_kron_real,
             Xpilotreal_: batch_Xpilot_real, Xpilotimag_: batch_Xpilot_imag, HH_: HH_vec_, Xdata_: batch_Xdata,
             Ydata_: batch_Ydata})
     
        _loss.append(b_loss)
        s_conste = np.zeros(X_data_hat_.shape, dtype=float)
        Batch = X_data_hat_.shape[0]
        row = X_data_hat_.shape[1]
        col = X_data_hat_.shape[2]
        #        print(row)
        #        print(col)
        x_bit_hat_ = []
        x_bit_true_ = []
        for bb in range(Batch):
            for ii in range(row):  # 注意range的写法
                for jj in range(col):
                    for kk in range(len(S_QPSK)):
                        #                     print(len(S_16QAM))
                        D[kk] = np.abs(X_data_hat_[bb, ii, jj] - S_QPSK[kk])
                    index = np.argmin(D)
                    s_conste[bb, ii, jj] = S_QPSK[index]
        #    print(s_conste.shape)
        x_conste = s_conste[:, 0:N, :] + 1j * s_conste[:, N:2 * N, :]  # 估计出来的符号值
        #        print(x_conste)
        #        print(x_complex_.shape)
        Batch1 = x_conste.shape[0]
        row1 = x_conste.shape[1]
        col1 = x_conste.shape[2]
        for bb in range(Batch1):
            for ii in range(row1):
                for jj in range(col1):
                    x_bit_hat = demapping_table_QPSK[x_conste[bb, ii, jj]]  # 这是一个tuple
                    x_bit_true = demapping_table_QPSK[x_complex_[bb, ii, jj]]  # 这是一个tuple
                    #               print(np.array(x_bit_hat).shape) #只有array有shape
                    x_bit_hat_.append(x_bit_hat)
                    x_bit_true_.append(x_bit_true)

        bit_hat_ = np.array(x_bit_hat_)
        bit_true_ = np.array(x_bit_true_)

        bit_hat = bit_hat_.reshape(mu * N * batch_size * Td, 1)
        bit_true = bit_true_.reshape(mu * N * batch_size * Td, 1)

        errbit_temp = (bit_hat != bit_true)
        errbit = np.sum(errbit_temp)
        ber = np.mean(errbit_temp)
        _ber.append(ber)
        #             print(x_bit_hat)
        error_symbol = np.not_equal(x_complex_, x_conste).astype('float64')
        SER = np.mean(error_symbol)
        SER_train.append(SER)

    return np.mean(_loss), np.mean(SER_train), np.mean(_ber), x_conste, batch_Xpilot_real, batch_Xpilot_imag, HH_est_,HH_sample,batch_Ydata, batch_Xdata, MSE_


def Valid_batch(sess):
    _loss = []
    SER_valid = []
    _ber = []
    packet = valid_size // batch_size
    for offset in range(packet):
        batch_Ypilot, batch_YJCD, Xpilot_kron, Xpilot_kron_real, batch_Xpilot_real, batch_Xpilot_imag, batch_Ydata, batch_Xdata, HH_sample, HH_vec_, HH_real_, x_complex_ = generate_data_iid_test(
            batch_size, M, N, snrdb_train)
        b_loss, X_data_hat_, Xpilot_real_equal_, dampingCE_, gamma_, gamma1_, corre_, expec_ = sess.run(
            [cost, X_Ouput, Xpilot_real_equal, dampingCE, gamma, gamma1, corre, expec],
            {Ypilot_vec_: batch_Ypilot, Ypilot_JCD_: batch_YJCD, Xpilotkron_real: Xpilot_kron_real,
             Xpilotreal_: batch_Xpilot_real, Xpilotimag_: batch_Xpilot_imag, HH_: HH_vec_, Xdata_: batch_Xdata,
             Ydata_: batch_Ydata})
        _loss.append(b_loss)
        s_conste = np.zeros(X_data_hat_.shape, dtype=float)
        Batch = X_data_hat_.shape[0]
        row = X_data_hat_.shape[1]
        col = X_data_hat_.shape[2]
        x_bit_hat_ = []
        x_bit_true_ = []
        for bb in range(Batch):
            for ii in range(row):  # 注意range的写法
                for jj in range(col):
                    for kk in range(len(S_QPSK)):
                        #                     print(len(S_16QAM))
                        D[kk] = np.abs(X_data_hat_[bb, ii, jj] - S_QPSK[kk])
                    index = np.argmin(D)
                    s_conste[bb, ii, jj] = S_QPSK[index]
        #    print(s_conste.shape)
        x_conste = s_conste[:, 0:N, :] + 1j * s_conste[:, N:2 * N, :]

        Batch1 = x_conste.shape[0]
        row1 = x_conste.shape[1]
        col1 = x_conste.shape[2]
        for bb in range(Batch1):
            for ii in range(row1):
                for jj in range(col1):
                    x_bit_hat = demapping_table_QPSK[x_conste[bb, ii, jj]]  # 这是一个tuple
                    x_bit_true = demapping_table_QPSK[x_complex_[bb, ii, jj]]  # 这是一个tuple
                    #               print(np.array(x_bit_hat).shape) #只有array有shape
                    x_bit_hat_.append(x_bit_hat)
                    x_bit_true_.append(x_bit_true)

        bit_hat_ = np.array(x_bit_hat_)
        bit_true_ = np.array(x_bit_true_)

        bit_hat = bit_hat_.reshape(mu * N * batch_size * Td, 1)
        bit_true = bit_true_.reshape(mu * N * batch_size * Td, 1)

        errbit_temp = (bit_hat != bit_true)
        errbit = np.sum(errbit_temp)
        ber = np.mean(errbit_temp)
        _ber.append(ber)
        #             print(x_bit_hat)
        error_symbol = np.not_equal(x_complex_, x_conste).astype(float)
        SER = np.mean(error_symbol)
        SER_valid.append(SER)

    return np.mean(_loss), np.mean(SER_valid), np.mean(_ber), x_conste, dampingCE_, gamma_, gamma1_, corre_, expec_


def Train():
    print("\nTraining ...")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        weight_file = weight_mat + 'oamp.mat'
        best_valid_loss = 0
        for i in range(epochs):
            #  change the learning rate in here
            start_time = time.time()
            train_loss, ser_train, ber_train, x_hat_train, batch_Xpilot_real, batch_Xpilot_imag, HH_est_, HH_sample, batch_Ydata, batch_Xdata, MSE_ = Train_batch(
                sess)
            valid_loss, ser_valid, ber_valid, x_hat_valid, dampingCE_, gamma_, gamma1_, corre_, expec_ = Valid_batch(sess)
            print(gamma_)
            print(gamma1_)
            print(corre_)
            print(expec_)
            time_taken = time.time() - start_time
            print("Epoch %d Valid Loss: %.6f, Valid SER: %.6f, Valid BER: %.6f, Time Cost: %.2f s"
                  % (i + 1, valid_loss, ser_valid, ber_valid, time_taken))
            if i == 0 or (i > 0 and valid_loss < best_valid_loss):
                best_valid_loss = valid_loss  # 保存最优的网络loss
                #        print(best_valid_loss)
                Save(weight_file)
        print("\nTraining is finished.")
    return x_hat_train, x_hat_valid, batch_Xpilot_real, batch_Xpilot_imag, HH_est_, HH_sample, batch_Ydata, batch_Xdata, MSE_


def Test_batch(sess, ebn0_test):
    _loss = []
    SER_test = []
    sernum = 0
    _ber = []
    SER_test = []
    x_bit_hat_ = []
    x_bit_true_ = []
    while sernum < errsum:
        batch_Ypilot, batch_YJCD, Xpilot_kron, Xpilot_kron_real, batch_Xpilot_real, batch_Xpilot_imag, batch_Ydata, batch_Xdata, HH_sample, HH_vec_, HH_real_, x_complex_ = generate_data_iid_test(
            batch_size, M, N, snrdb_train)
        b_loss, X_data_hat_, Xpilot_real_equal_, damping_CE, MMSE_, gamma_, gamma1_, corre_, expec_ = sess.run(
            [cost, X_Ouput, Xpilot_real_equal, dampingCE, MMSE,gamma, gamma1, corre, expec],
            {Ypilot_vec_: batch_Ypilot, Ypilot_JCD_: batch_YJCD, Xpilotkron_real: Xpilot_kron_real,
             Xpilotreal_: batch_Xpilot_real, Xpilotimag_: batch_Xpilot_imag, HH_: HH_vec_, Xdata_: batch_Xdata,
             Ydata_: batch_Ydata})
        _loss.append(b_loss)
        s_hat_test = np.zeros(X_data_hat_.shape, dtype=float)

        Batch = X_data_hat_.shape[0]
        row = X_data_hat_.shape[1]
        col = X_data_hat_.shape[2]
        for bb in range(Batch):
            for ii in range(row):  # 注意range的写法
                for jj in range(col):
                    for kk in range(len(S_QPSK)):
                        D[kk] = np.abs(X_data_hat_[bb, ii, jj] - S_QPSK[kk])
                    index = np.argmin(D)
                    s_hat_test[bb, ii, jj] = S_QPSK[index]

        x_conste = s_hat_test[:, 0:N, :] + 1j * s_hat_test[:, N:2 * N, :]
        Batch1 = x_conste.shape[0]
        row1 = x_conste.shape[1]
        col1 = x_conste.shape[2]
        for bb in range(Batch1):
            for ii in range(row1):
                for jj in range(col1):
                    x_bit_hat = demapping_table_QPSK[x_conste[bb, ii, jj]]  # 这是一个tuple
                    x_bit_true = demapping_table_QPSK[x_complex_[bb, ii, jj]]  # 这是一个tuple
                    #               print(np.array(x_bit_hat).shape) #只有array有shape
                    x_bit_hat_.append(x_bit_hat)
                    x_bit_true_.append(x_bit_true)

        bit_hat_ = np.array(x_bit_hat_)
        bit_true_ = np.array(x_bit_true_)

        errbit_temp = (bit_hat_ != bit_true_)
        errbit = np.sum(errbit_temp)
        ber = np.mean(errbit_temp)
        _ber.append(ber)
        #             print(x_bit_hat)
        error_symbol = np.not_equal(x_complex_, x_conste).astype(float)

        errnum = np.sum(error_symbol)
        sernum = errnum + sernum

        SER = np.mean(error_symbol)
        SER_test.append(SER)
        _loss.append(b_loss)

    return np.mean(_loss), np.mean(SER_test), np.mean(_ber), damping_CE, x_conste, MMSE_, gamma_, gamma1_, corre_, expec_


def Test(snr_test):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(update)
        loss_test, ser_test, ber_test, damping_CE, x_hat_test, MMSE_, gamma_, gamma1_, corre_, expec_ = Test_batch(sess, snr_test)
        print(gamma_)
        print(gamma1_)
        print(corre_)
        print(expec_)
        print("Eb/N0: %.0f, Testing Loss: %.6f, Testing SER: %.6f, Testing BER: %.6f" % (
        snr_test, loss_test, ser_test, ber_test))
    return loss_test, ser_test, ber_test, damping_CE, x_hat_test, gamma_, gamma1_, corre_, expec_


def Save(weight_file):
    dict_name = {}
    for varable in tf.trainable_variables():
        dict_name[varable.name] = varable.eval()
    sc.savemat(weight_file, dict_name)

with tf.Graph().as_default():
    
    with tf.variable_scope('gamma'):
        gamma=Variable((itermax,))
        
    with tf.variable_scope('gamma1'):
        gamma1=Variable1((itermax,))  
        
    with tf.variable_scope('corre'):
        corre=Variable2((itermax,)) 
        
    with tf.variable_scope('expec'):
        expec=Variable3((itermax,))  
        
#    W1 =  tf.Variable(tf.ones(iter_JCD, dtype="float64"),name="damping",dtype="float64")  
#    W2 =  tf.Variable(tf.zeros(iter_JCD, dtype="float64"),name="damping",dtype="float64")     
#    W3 =  tf.Variable(tf.ones(iter_JCD, dtype="float64"),name="damping",dtype="float64")      
#    W4 =  tf.Variable(tf.zeross(iter_JCD, dtype="float64"),name="damping",dtype="float64")      

    #  damping = tf.Variable(tf.ones(5,dtype="float64"),name="damping",dtype="float64")
    Xpilotreal_ = tf.placeholder(dtype='float64', shape=[None, N, Tp])  # The real part of the pilot
    Xpilotimag_ = tf.placeholder(dtype='float64', shape=[None, N, Tp])  # The imaginary part of the pilot
    # Xpilotkron_= tf.placeholder(dtype='float64',shape=[None,M*N,M*Tp])
    Xpilotkron_real = tf.placeholder(dtype='float64', shape=[None, 2 * M * Tp, 2 * M * N])  # The kron pilot (real)
    HH_ = tf.placeholder(dtype='float64', shape=[None, 2 * M * N, 1])  # The vector of channel (real)
    Xdata_ = tf.placeholder(dtype='float64', shape=[None, 2 * N, Td])  # The data matrix (real)
    Ypilot_JCD_ = tf.placeholder(dtype='float64', shape=[None, 2 * M * Tc, 1])  # Y_  joint pilot and data
    Ypilot_vec_ = tf.placeholder(dtype='float64', shape=[None, 2 * M * Tp, 1])  # Y_  pilot
    Ydata_ = tf.placeholder(dtype='float64', shape=[None, 2 * M, Td])  # Y_  data
    Id = tf.eye(2 * M * N, batch_shape=[batch_size], dtype='float64')  # identity matrix
    IN = tf.eye(2 * N, batch_shape=[batch_size], dtype='float64')  # identity matrix
    Ic = tf.eye(M, dtype='float64')  # identity matrix
    # batch_size = tf.placeholder(tf.float64)
#    v_B2A_inv1 = tf.ones((batch_size, 2 * M, 1), dtype='float64')  # Initialization of the v_B2A_inv
#    x_B2A1 = tf.zeros((batch_size, 2 * M, 1), dtype='float64')  # Initialization of the x_B2A
    #  t=tf.zeros([],dtype='float64')
    noise_var = Sigma2
    eps = 5e-13 * tf.ones(1, dtype='float64')
    eps1 = 5e-7 * tf.ones(1, dtype='float64')
    dampingCE = 0.2 * tf.ones(1, dtype='float64')
    Xpilot_equal = Xpilotkron_real
    Ypilot_equal = Ypilot_vec_
    loss_sum = tf.zeros(1, dtype='float64')
    mse_sum = tf.zeros(1, dtype='float64')
    MMSE = list()
    #  HH_est = tf.zeros((batch_size,2*M,2*N), dtype='float64')
    Cov_noise_CE = Sigma2 * tf.eye(2 * M * Tp, batch_shape=[batch_size], dtype='float64') / 2
    Cov_noise_SD = Sigma2 * tf.eye(2 * M * Tp, batch_shape=[batch_size], dtype='float64') / 2

    zero_mtx_CE = tf.zeros((batch_size, M, M), dtype='float64')  # Initialization of the x_B2A
    zero_mtx_SD = tf.zeros((batch_size, M * Tc, M * Tc), dtype='float64')  # Initialization of the x_B2A
    Ones_vec = tf.ones((batch_size, 1, Tp), dtype='float64')  # Initialization of the v_B2A_inv  

    for kk in range(iter_JCD):  # Iteration of JCD
        
        print(kk)

        X_data_hat_mtx = tf.zeros((batch_size, 2 * N, 1), dtype=tf.float64)
        Var_data_mtx = tf.zeros((batch_size, 2 * N, 1), dtype=tf.float64)
        v_B2A_inv1 = tf.ones((batch_size, 2 * M * N, 1), dtype='float64')
        x_B2A1 = tf.zeros((batch_size, 2 * M * N, 1), dtype='float64')
        xhat_init = 0 * tf.ones((1), dtype='float64')  # Initial mean of channel h
        xvar_init = 1 * tf.ones((1), dtype='float64') / N  # Initial Covariance of channel h

        for tt in range(itermax):  # tf.expand_dims, tf.tile, tf.squeeze 等操作

            gram1 = tf.matmul(tf.matmul(Xpilot_equal, tf.matrix_inverse(Cov_noise_CE), adjoint_a=True), Xpilot_equal)
#            print(gram1)
#            print(tf.matrix_diag(tf.squeeze(v_B2A_inv1)))
            V1 = tf.matrix_inverse(tf.matrix_diag(tf.squeeze(v_B2A_inv1)) + gram1)

            ZZ1 = tf.matmul(tf.matmul(Xpilot_equal, tf.matrix_inverse(Cov_noise_CE), adjoint_a=True), Ypilot_equal) + x_B2A1  # pay attention to x_b2a

            mu_x1 = tf.matmul(V1, ZZ1)
            #       print(mu_x)
            diagV1 = tf.expand_dims(tf.real(tf.matrix_diag_part(V1)), -1)
            #      diagV = tf.expand_dims( noise_var*tf.real(tf.matrix_diag_part(V)),-1 )
            v_A2B1 = tf.maximum(tf.divide(diagV1, 1 - tf.multiply(diagV1, v_B2A_inv1)), eps1)
            #       print(v_A2B)
            x_A2B1 = tf.multiply(v_A2B1, tf.divide(mu_x1, diagV1) - x_B2A1)
            #       x_B,v_B = MMSE_Constellation(x_A2B, tf.clip_by_value(tf.real(v_A2B),eps,1/eps))
            x_B1, v_B1 = MMSE_AWGN_Gaussian(x_A2B1, v_A2B1, xhat_init, xvar_init / 2)

            v_H_mean1 = tf.maximum(v_B1, eps1)

            v_B2A_inv_new1 = tf.divide(tf.divide(v_A2B1 - v_H_mean1, v_H_mean1), v_A2B1)

            x_B2A_new1 = tf.divide(tf.divide((tf.multiply(x_B1, v_A2B1) - tf.multiply(x_A2B1, v_H_mean1)), v_H_mean1), v_A2B1)

            v_B2A_inv1 = (1 - tf.sigmoid(dampingCE)) * v_B2A_inv1 + tf.sigmoid(dampingCE) * v_B2A_inv_new1
            x_B2A1 = (1 - tf.sigmoid(dampingCE)) * x_B2A1 + tf.sigmoid(dampingCE) * x_B2A_new1            
#            v_B2A_inv1 = (1-damping[tt])*v_B2A_inv1+damping*v_B2A_inv_new1
#            print(v_B2A_inv1)
#            x_B2A1 = (1-damping)*x_B2A1+damping*x_B2A_new1            
        MSE = tf.nn.l2_loss(x_B1 - HH_)
        
        MMSE.append(tf.nn.l2_loss(x_B1 - HH_ ))
        
        H_real = tf.transpose(tf.reshape(x_B1[:, 0:M * N, :], [batch_size, N, M]), [0, 2, 1])
        #      print(H_real)
        H_imag = tf.transpose(tf.reshape(x_B1[:, M * N:2 * M * N, ], [batch_size, N, M]), [0, 2, 1])
        #      print(H_imag)
        HH_est = tf.concat([tf.concat([H_real, H_imag], 1), tf.concat([tf.negative(H_imag), H_real], 1)], 2)
        #    print(v_H_mean)
        Cov_H_vec = v_H_mean1[:, 0:M * N, :] + v_H_mean1[:, M * N:2 * M * N, :]
        #    print(Cov_H_vec)
        Cov_H = tf.transpose(tf.reshape(Cov_H_vec, [batch_size, N, M]), [0, 2, 1])  # Python reshape 和 matlab 相反
        #    print(Cov_H)
        # Sigma2_h = tf.transpose(tf.reduce_sum(Cov_H, 2, keepdims=True) + noise_var,[0,2,1])  # Python reshape 和 matlab sum相反
        Sigma2_h = tf.reduce_sum(Cov_H, 2, keepdims=True) + noise_var
#        print(Sigma2_h)
        Cov_H_real = tf.concat([tf.concat([tf.matrix_diag(tf.squeeze(Sigma2_h, [2] )) / 2, zero_mtx_CE], 1),
                                tf.concat([zero_mtx_CE, tf.matrix_diag(tf.squeeze(Sigma2_h, [2] )) / 2], 1)], 2)
        print(Cov_H_real)
        
        for ii in range(Td):
            
            v2 = tf.ones((batch_size,), dtype='float64')
            
            I=tf.eye(2*M,batch_shape=[batch_size],dtype='float64')
            
            s=tf.zeros((batch_size,2*N,1),dtype=tf.float64)#给s多加一个维度
            
            for t in range(itermax):
        
                v2=tf.tile(tf.expand_dims(tf.expand_dims(v2, axis=-1), axis=-1),[1,2*M,2*N])  #弄成一个矩阵去乘
 #               print(v2)
                RR=tf.matrix_inverse(tf.multiply(v2, tf.matmul(HH_est, HH_est, adjoint_b = True)) + Cov_H_real)
#        print(RR)
                W=tf.multiply(v2, tf.matmul(HH_est,RR,adjoint_a = True))  #LMMSE Matrix
#        W=tf.matmul(v2*A_,tf.matrix_inverse(tf.matmul(v2*A_,A_,adjoint_b = True)+sigma2*tf.eye(N)),adjoint_a = True)  #LMMSE Matrix
#        W=tf.matmul(A_,tf.matrix_inverse(tf.matmul(A_,A_,adjoint_b = True)),adjoint_a = True)
#        W=tf.matrix_transpose(A_)
                tr=tf.trace(tf.matmul(W,HH_est))
    #    print(tr)
                tr=tf.tile(tf.expand_dims(tf.expand_dims(tr, axis=-1), axis=-1),[1,2*M,2*N])
        
                W=2*N/tr*W                 
#                print(tf.expand_dims(Ydata_[:,:,ii],-1).shape)
#                print(s.shape)                
                z = tf.expand_dims(Ydata_[:,:,ii],-1)-tf.matmul(HH_est,s)
                
                r = s + gamma[t]*tf.matmul(W,z)
#        print(r)
                noise_equal = tf.squeeze(tf.reduce_sum(Cov_H_real, [1, 2], keep_dims=True))
#                print(noise_equal)
                v2 = tf.maximum(tf.div(tf.norm(z, axis=[-2,-1])**2-noise_equal,tf.trace(tf.matmul(HH_est,HH_est,adjoint_a=True))), eps)
#        tau2 = v2/(2*N)*(2*N+(gamma[t]**2-2*gamma[t])*2*M)+gamma[t]**2*sigma2/(2*N)*tf.trace(tf.matmul(W,W,adjoint_b=True))
#        tau2=M/(M-N)*sigma2*tf.ones((batch_size,))
#                print(v2)
                
                B=I-gamma1[t]*tf.matmul(W, HH_est)
                
                tau2 = v2/2/N*tf.trace(tf.matmul(B,B,adjoint_b=True))+gamma1[t]*gamma1[t]*noise_equal/2/M/4/N*tf.trace(tf.matmul(W,W,adjoint_b=True))
#                print(r)
#                print(tau2)
                s = MMSE_Constellation(r,tf.tile(tf.expand_dims(tf.expand_dims(tau2, axis=-1), axis=-1),[1,2*M,1]) ) 
#                print(s)
                s=corre[t]*(s- expec[t]*r) 
#                print(s)
#                v_B2A_inv = (1 - damping) * v_B2A_inv + damping*v_B2A_inv_new
#                x_B2A = (1 - damping) * x_B2A + damping * x_B2A_new    
            X_data_hat_mtx = tf.concat([X_data_hat_mtx, s], 2)
            Var_data_mtx = tf.concat([Var_data_mtx, tf.tile(tf.expand_dims(tf.expand_dims(tau2, axis=-1), axis=-1),[1,2*M,1])], 2)

        #      print(X_data_hat_mtx[:,0:N,1:Td+1])
        #     print(Var_data_mtx[:, 0:N, 1:Td + 1])
        Xpilot_real_equal = tf.concat([Xpilotreal_, X_data_hat_mtx[:, 0:N, 1:Td + 1]], 2)
        Xpilot_imag_equal = tf.concat([Xpilotimag_, X_data_hat_mtx[:, N:2 * N, 1:Td + 1]], 2)
#        Xpilot_real_equal = tf.concat([W1[tt]*(Xpilotreal_), W2[tt]*(X_data_hat_mtx[:, 0:N, 1:Td + 1])], 2)
#        Xpilot_imag_equal = tf.concat([W3[tt]*(Xpilotimag_), W4[tt]*(X_data_hat_mtx[:, N:2 * N, 1:Td + 1])], 2)
        #  print(Xpilot_real_equal)
        #  print(Xpilot_imag_equal)
        XJCD_kron_real = tf.py_func(np.kron, [tf.transpose(Xpilot_real_equal, [0, 2, 1]), Ic], tf.float64)
        XJCD_kron_imag = tf.py_func(np.kron, [tf.transpose(Xpilot_imag_equal, [0, 2, 1]), Ic], tf.float64)
        #  print(XJCD_kron_real)
        Xpilot_equal = tf.concat([tf.concat([XJCD_kron_real, XJCD_kron_imag], 1),
                                  tf.concat([tf.negative(XJCD_kron_imag), XJCD_kron_real], 1)], 2)
    
        Ypilot_equal = Ypilot_JCD_
        
        X_Ouput = X_data_hat_mtx[:, :, 1:Td + 1]
        
        X_var = Var_data_mtx[:, :, 1:Td + 1]
#        print(Var_data_mtx)
        Var_data = X_var[:, 0:N, :] + X_var[:, N:2 * N, :]
#        print(Var_data)
        Sigma2_data = tf.reduce_sum(Var_data, 1, keepdims=True) / N + noise_var
 #       print(Sigma2_data)
        Noise_init = noise_var * Ones_vec
#        print(Noise_init)
        Sigma_diag = tf.concat([Noise_init, Sigma2_data], 2)
#        print(Sigma_diag)
        Cov_data = tf.matrix_diag(tf.squeeze(Sigma_diag,[1]))
#        print(Cov_data)
        Cov_data_mtx = tf.py_func(np.kron, [Cov_data, Ic], tf.float64)
        #   print(Cov_data_mtx)
        Cov_noise_CE = tf.concat(
            [tf.concat([Cov_data_mtx / 2, zero_mtx_SD], 1), tf.concat([zero_mtx_SD, Cov_data_mtx / 2], 1)], 2)
#        print(Cov_noise_CE)
        cost = tf.nn.l2_loss(Xdata_ - X_Ouput)  # l2 loss function
        #    print(cost)
        loss_sum = tf.add(loss_sum, cost)
        mse_sum = tf.add(mse_sum, MSE)
        
    cost_sum = tf.add(mse_sum, loss_sum)          
    #  print(loss_sum)
    #  X_Ouput = X_data_hat_mtx[:, :, 1:Td + 1]
#      cost = tf.nn.l2_loss(Xdata_ - X_Ouput)  # l2 loss function
#    print(cost)
#    print(MSE)
    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost_sum)
    #    Training DetNet
    x_hat_train_, x_hat_valid_, batch_Xpilot_real, batch_Xpilot_imag, HH_hat_, HH_sample, batch_Ydata, batch_Xdata,MSE_Out= Train()

    update = Network_ini(sc.loadmat(weight_mat + 'oamp.mat'))

#    print(update)

    loss_test, ser_test, ber_test,  damping_CE, x_hat_test, gamma_, gamma1_, corre_, expec_ = Test(snr_train)





