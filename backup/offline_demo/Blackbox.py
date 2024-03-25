#!/usr/bin/python3

import os
import sys
from sys import stdin
import numpy as np

def safe_print(n):
    print(n)
    sys.stdout.flush() # Please flush the output for each print, otherwise it will result in a Time Limit Exceeded!


def read_inputdata(input_data):
    m = input_data.split(' ')
    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        m[ii] = float(m[ii])
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j
    return n

def whole_system(DL0_bf,DL1_bf):
    global data_para
    DL0_freq = np.zeros(shape=(int(300), 32), dtype='complex128')
    bf_tmp = DL0_bf


    bfsize0 = bf_tmp.shape[0]


    DL0_datatmp_local = data_para['DL0_datatmp'][:,0:bfsize0]
    DL0_datatmp_local = np.mat(DL0_datatmp_local)
    DL0_freq = np.sqrt(32)*1/np.sqrt(bfsize0)*DL0_datatmp_local*np.conj(bf_tmp)  
    
    DL1_freq = np.zeros(shape=(int(300), 32), dtype='complex128')
    bf_tmp = DL1_bf

    bfsize1 = bf_tmp.shape[0]

    
    DL1_datatmp_local = data_para['DL1_datatmp'][:,0:bfsize1]
    DL1_datatmp_local = np.mat(DL1_datatmp_local)
    DL1_freq = np.sqrt(32)*1/np.sqrt(bfsize1)*DL1_datatmp_local*np.conj(bf_tmp)  

    sig0_signal = DL0_freq*data_para['DL_ch_set_DL0']
    sig1_signal = DL1_freq*data_para['DL_ch_set_DL1']

    UL_signal_rate = np.multiply(np.multiply(np.multiply(sig0_signal, sig0_signal), np.conj(sig1_signal)),data_para['calib_c'])    
        
    rx_signal_freq = np.zeros(shape=(UL_signal_rate.shape[0],32,UL_signal_rate.shape[1]), dtype = "complex128")
    for ii in range(UL_signal_rate.shape[1]):
        rx_signal_freq[:,:,ii] = UL_signal_rate[:,ii]*data_para['DL_ch_set_UL'][:,ii].T
    rx_signal_tot_freq = UL_signal_rate * data_para['DL_ch_set_UL'].T
    
    return rx_signal_tot_freq


def exit_with_wrong_answer(message):
    print(f'Wrong Answer: {message}', file=sys.stderr)
    exit()

def exit_with_judge_error(message):
    print(f'Internal error: {message}', file=sys.stderr)
    exit(1)


def accept_with_score(score):
    print(f'Accepted! Score: {score}', file=sys.stderr)
    exit()

def calc_score(input_data):
    global data_para,signal_num_targ
    try:
        m = input_data.strip().split(' ')#list ['XXXX','XXXX',...]  len 128
    except Exception as e:
        exit_with_wrong_answer(f'Protocol error, last write in score was {line}')

    if not np.mod(len(m),32) == 0:
        exit_with_wrong_answer(f'Protocol error')

    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        try:
            m[ii] = float(m[ii])
        except Exception as e:
            exit_with_wrong_answer(f'Protocol error, could not read')
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j

    signal_dl0channel = np.mat(n)
    if not signal_num_targ == int(signal_dl0channel.shape[1]/32):
        exit_with_wrong_answer(f'Invalid result input matrix!')
    signal_dl0channel = signal_dl0channel.reshape(int(signal_num_targ),32)

    for ii in range(int(signal_num_targ)):
        if not 0.99 <= np.linalg.norm(signal_dl0channel[ii,:]) <= 1.01:
            exit_with_wrong_answer(f'Invalid data in input weight!')


    dl_channel = signal_dl0channel.T
    dl_Nullproj = np.eye(32) - dl_channel*np.linalg.inv(dl_channel.H*dl_channel)*dl_channel.H
    if signal_num_targ == 2:
        after_proj_power = dl_Nullproj*data_para['DL_ch_set_DL0']
    else:
        exit_with_judge_error(f'Invalid signal num : {signal_num}')
    result = 0

    for ii in range(signal_num_targ):
        result_tmp = np.linalg.norm(after_proj_power[:,ii])**2
        result = result + result_tmp

    result = 15*(1 - result/signal_num_targ)
    result = np.round(result,decimals=6)
    accept_with_score(result) #The submission code need this line
    return result


def blackboxSystem(input_1,input_2):
    global data_para, signal_num_targ
    try:
        input_in = open(sys.argv[1], 'r').read().strip()
        split_data = input_in.split('\n') #Split data by \n
        case = int(split_data[0]) # Case index
        if not 1 <= case <= 4:
            exit_with_judge_error(f'Invalid data in input file: {case}')
        #Read preloaded data in 1.in
        dataDL0 = read_inputdata(split_data[2])
        dataDL1 = read_inputdata(split_data[4])
        dataUL = read_inputdata(split_data[6])
        data_calib = read_inputdata(split_data[8])
        data_tmpDL0 = read_inputdata(split_data[10])
        data_tmpDL1 = read_inputdata(split_data[12]) 
        signal_num = int(len(dataDL0)/32)
        # Reshape data to matrices
        data_para = {}
        data_para['DL_ch_set_DL0'] = np.mat(dataDL0.reshape(signal_num,32)).T
        data_para['DL_ch_set_DL1'] = np.mat(dataDL1.reshape(signal_num,32)).T
        data_para['DL_ch_set_UL'] = np.mat(dataUL.reshape(signal_num,32)).T
        data_para['calib_c'] = np.mat(data_calib.reshape(50,300)).T
        data_para['DL0_datatmp'] = np.mat(data_tmpDL0.reshape(32,300)).T
        data_para['DL1_datatmp'] = np.mat(data_tmpDL1.reshape(32,300)).T
    except Exception as e:
        exit_with_judge_error(f'Invalid data in input file')

    m = [int(x) for x in open(sys.argv[2], 'r').read().strip().split(' ')]
    if not 0 <= int(m[0]) <= 200:
        exit_with_judge_error(f'Invalid data in ans file: {m}')
    for ii in range(signal_num):
        m[ii] = int(m[ii])
    m = np.array(m)

    calib_c = data_para['calib_c'][:,:signal_num]
    data_para['calib_c'] = calib_c*np.diag(m)
    if signal_num == 2:
        signal_num_targ = 2
    ul_channel_all = []
    dl0_channel_all = []
    ul_Nullproj = []

    score = 0
    try:
        m = input_1.strip().split(' ')#Split by ' '
        if m[0] == 'END': # When program end, calculate score
            safe_print('Roger that') # Tell submission program, ready to receive estimated data now
            calc_score() # Calculate score
    except Exception as e:
        exit_with_wrong_answer(f'Protocol error, last write was {line}')
    if not np.mod(len(m),32) == 0:
        exit_with_wrong_answer(f'Protocol error')
    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        try:
            m[ii] = float(m[ii])
        except Exception as e:
            exit_with_wrong_answer(f'Protocol error, could not read')
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j

    input_weight_mtx1 = np.mat(n)

    stream_num1 = input_weight_mtx1.shape[1]/32
    input_weight_mtx1 = input_weight_mtx1.reshape(int(stream_num1),32)

    if not 0.999 <= np.linalg.norm(input_weight_mtx1)**2 <= 1.001:
        exit_with_wrong_answer(f'Invalid data in input weight 1!')


    try:
        m = input_2.strip().split(' ')
    except Exception as e:
        exit_with_wrong_answer(f'Protocol error, last write was {line}')
    if not np.mod(len(m),32) == 0:
        exit_with_wrong_answer(f'Protocol error')

    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        try:
            m[ii] = float(m[ii])
        except Exception as e:
            exit_with_wrong_answer(f'Protocol error, could not read')
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j

    input_weight_mtx2 = np.mat(n)

    stream_num2 = input_weight_mtx2.shape[1]/32
    input_weight_mtx2 = input_weight_mtx2.reshape(int(stream_num2),32)

    if not 0.999 <= np.linalg.norm(input_weight_mtx2)**2 <= 1.001:
        exit_with_wrong_answer(f'Invalid data in input weight 2!')

    DL0_bf = input_weight_mtx1
    DL1_bf = input_weight_mtx2

    rx_signal_tot_freq = whole_system(DL0_bf,DL1_bf) # output matrix to submission program. 300X32.

    rx_signal_tot_freq = np.mat(rx_signal_tot_freq)
    rx_signal_tot_freq = rx_signal_tot_freq.T

    input_data_ravel = rx_signal_tot_freq.ravel(order="F") # Convert matrix to a vector
    input_data_ravel = np.round(input_data_ravel,decimals=6) # 6 decimals float

    output = ''
    for ii in range(input_data_ravel.shape[1]):
        if ii == input_data_ravel.shape[1]-1:
            m = str(np.real(input_data_ravel[0,ii])) + ' ' + str(np.imag(input_data_ravel[0,ii]))
        else:
            m = str(np.real(input_data_ravel[0,ii])) + ' ' + str(np.imag(input_data_ravel[0,ii])) + ' '
        output = output + m
    return output
    #safe_print(output)
