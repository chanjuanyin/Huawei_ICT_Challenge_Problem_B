#!/usr/bin/python3
import numpy as np
import sys
from sys import stdin
import Blackbox as blk


def safe_print(n):
    print(n)
    sys.stdout.flush() # Please flush the output for each print, otherwise it will result in a Time Limit Exceeded!


def bf_norm(bf_new):
    bf_new = bf_new/np.linalg.norm(bf_new)
    return bf_new

def bf_norm_multiple_stream(bf_new):
    bf_norm = bf_new/np.linalg.norm(bf_new)
    return bf_norm

def norm_multiple_stream_result(L_est):
    stream_num = L_est.shape[0]
    for ii in range(stream_num):
        L_est[ii,:] = bf_norm(L_est[ii,:])
    return L_est

def mtx2outputdata(input_data):
    stream_num = input_data.shape[1] # Example input_data matrix size: 32 x N_stream.
    input_data_ravel = input_data.ravel(order="F") # matrix to vector
    input_data_ravel = np.round(input_data_ravel,decimals=6) # 6 decimals float
    
    output = ''
    for ii in range(input_data_ravel.shape[1]):
        if ii == input_data_ravel.shape[1]-1:
            m = str(np.real(input_data_ravel[0,ii])) + ' ' + str(np.imag(input_data_ravel[0,ii])) # Example: [1+2j,1.4+3.1j] -> ['1 2 1.4 3.1']
        else:
            m = str(np.real(input_data_ravel[0,ii])) + ' ' + str(np.imag(input_data_ravel[0,ii])) + ' '
        output = output + m
    # safe_print(output)
    return output

def mtx2outputdata_result(input_data):
    stream_num = input_data.shape[0] # Example input_data matrix size: N_target X 32.
    input_data = input_data.T
    input_data_ravel = input_data.ravel(order="F") # matrix to vector
    input_data_ravel = np.round(input_data_ravel,decimals=6) # 6 decimals float
    
    output = ''
    for ii in range(input_data_ravel.shape[1]):
        if ii == input_data_ravel.shape[1]-1:
            m = str(np.real(input_data_ravel[0,ii])) + ' ' + str(np.imag(input_data_ravel[0,ii])) # Example: [1+2j,1.4+3.1j] -> ['1 2 1.4 3.1']
        else:
            m = str(np.real(input_data_ravel[0,ii])) + ' ' + str(np.imag(input_data_ravel[0,ii])) + ' '
        output = output + m
    # safe_print(output)
    return output

# def read_blackbox():
#     line = stdin.readline().strip()
#     m = line.split(' ')
#     complex_len = int(len(m)/2)
#     n = np.zeros(shape=(complex_len),dtype='complex128')
#     for ii in range(len(m)):
#         m[ii] = float(m[ii])
#     for ii in range(complex_len):
#         n[ii] = m[2*ii] + m[2*ii+1]*1j
#     n = n.reshape(300,32) # This step is to reshape Y to a matrix
#     n = n.T # This step is to match the size of Y in the document
#     return n

def read_blackbox(input_data):
    # line = stdin.readline().strip()
    m = input_data.split(' ')
    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        m[ii] = float(m[ii])
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j
    n = n.reshape(300,32) # This step is to reshape Y to a matrix
    n = n.T # This step is to match the size of Y in the document
    return n
def find_row_ratio():
    matrix_10= np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    matrix_10[0,0] = 1.0 + 0.0*1j
    matrix_10_input = mtx2outputdata(matrix_10)
    y_10 = blk.blackboxSystem(matrix_10_input, matrix_10_input)
    y_10 = read_blackbox(y_10)
    ratio_matrix = np.mat(np.zeros((32, 300), dtype=complex))
    ratio_matrix[0] = (y_10 / y_10)[0]
    
    for i in range(1, 32):
        matrix_01 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        matrix_01[0,i] = 1.0 + 0.0*1j
        matrix_01_input = mtx2outputdata(matrix_01)
        y_01 = blk.blackboxSystem(matrix_01_input, matrix_01_input)
        y_01 = read_blackbox(y_01)
        ratio_matrix[i] = (y_01 / y_10)[0]
    return ratio_matrix
# ---Step 1: Input 'Start'---
line1 = stdin.readline().strip()

# ---Step 1: Output W1 W2---
# N_stream1, N_stream2 can be [1,32]
N_stream1 = 32
N_stream2 = 32
# W1, W2 Example
W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
W2 = W1
W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
W1[0,0] = 1.0 + 0.0*1j
# W1 W2 Normalization
W1 = bf_norm_multiple_stream(W1)
W2 = bf_norm_multiple_stream(W2)
# Output W1 W2
input_01 = mtx2outputdata(W1)
input_02 = mtx2outputdata(W2)

# ---Step 3: Input Y---
rece_Y1 = blk.blackboxSystem(input_01, input_02)
Y = read_blackbox(rece_Y1)


def test_large(vector, order=0.1):
    return np.sum(np.abs(vector[0])) > order

N_tar=-1
h_idx = []
test_order = 5
if test_large(Y[0], order=test_order):
    h_idx.append(0)


for i in range(1, 32):
    W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    W1[i,0] = 1.0 + 0.0*1j
    W1 = bf_norm_multiple_stream(W1)
    W2 = W1
    # Output W1 W2
    input_01 = mtx2outputdata(W1)
    input_02 = mtx2outputdata(W2)
    rece_Y1 = blk.blackboxSystem(input_01, input_02)
    # Input Y
    Y01 = read_blackbox(rece_Y1)
    if test_large(Y01[i], order=test_order):
        h_idx.append(i)
    else: #not interference
        # print('haha')
        if N_tar != -1:
            if len(h_idx) >10:
                U, S, Vh = np.linalg.svd(Y01)
                tolerance = 1e-5 
                rank = np.sum(S > tolerance)
                N_tar = rank -1
            # assert rank == N_tar
            # assert np.linalg.matrix_rank(y_01) == N_tar
        else:
            print('here11')
            U, S, Vh = np.linalg.svd(Y01)
            tolerance = 1e-5 
            rank = np.sum(S > tolerance)
            N_tar = rank    
if N_tar == -1: #means case 3 case 4
    for i in range(1, 32):
        W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        W1[i,0] = 1.0 + 0.0*1j
        W1 = bf_norm_multiple_stream(W1)
        W2 = W1
        # Output W1 W2
        input_01 = mtx2outputdata(W1)
        input_02 = mtx2outputdata(W2)
        rece_Y1 = blk.blackboxSystem(input_01, input_02)
        # Input Y
        Y01 = read_blackbox(rece_Y1)
            
        
        if N_tar != -1:
            if len(h_idx) >10:
                U, S, Vh = np.linalg.svd(Y01)
                tolerance = 1e-5 
                rank = np.sum(S > tolerance)
                N_tar = rank -1
            # assert rank == N_tar
            # assert np.linalg.matrix_rank(y_01) == N_tar
        else:
            print('here11')
            U, S, Vh = np.linalg.svd(Y01)
            tolerance = 1e-5 
            rank = np.sum(S > tolerance)
            N_tar = rank            
# N_target can be [1,10]
N_target = N_tar # Notices that the N_target is different for each case. Contestants program need to estimate the number of target signals first.

rand_time = 50
np.random.seed(4518)
y_first_cols = []
W_matrices = []  # Store all W1 matrices
null = np.mat(np.zeros((N_target, 32))) + 1j * np.mat(np.zeros((N_target, 32)))

for j in range(rand_time):
    W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    W1[:, 0] = np.random.rand(32, 1) + 1j * np.random.rand(32, 1)
    W1 = bf_norm_multiple_stream(W1)
    W2 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    W2[0, 0] = 1.0 +  0.0*1j
    W2 = bf_norm_multiple_stream(W2)
    # Store the matrix for later retrieval
    W_matrices.append(W1)
    # Output W1 W2
    input_01 = mtx2outputdata(W1)
    input_02 = mtx2outputdata(W2)
    rece_Y1 = blk.blackboxSystem(input_01, input_02)    
    Y01 = read_blackbox(rece_Y1)
    for inter_idx in h_idx:  #delete the interference rows
        Y01[inter_idx, :] = 0
    # print("current Y01[inter_idx]= ", Y01[inter_idx])
    # Store the norm and the index
    y_first_cols.append((np.linalg.norm(Y01[:, 0]), j))
    # Sort by the norm and get indices of the smallest norms
y_first_cols_sort = sorted(y_first_cols, key=lambda x: x[0])
print('smallest_norm=',y_first_cols_sort)
chosen_indices = [idx for norm, idx in y_first_cols_sort[:N_target]]
# Retrieve corresponding W1 matrices
chosen_W_matrices = [np.array(W_matrices[idx][:,0]) for idx in chosen_indices]
print("length of chosen_W_matrices=", len(chosen_W_matrices))
ratio_matrix = find_row_ratio()
for k in range(len(chosen_W_matrices)):
    null_col = np.mat(np.array(ratio_matrix[:,0]) * chosen_W_matrices[k]).T
    null[k,:] = null_col
# print(null)
# L_est = np.mat((np.random.randn(N_target,32) + 1j*np.random.randn(N_target,32)))
L_est = null
# L_est Normalization
L_est = norm_multiple_stream_result(L_est)
#******Example code end*******


# ---Step 3: Output 'END' (if interaction end)---
safe_print('END')

# ---Step 5: Input 'Roger that'---
line2 = stdin.readline().strip()

# ---Step 5: Output L_est---

# Output L_est
L_est_str=  mtx2outputdata_result(L_est)
print("Number of signal = ",N_target)
print(h_idx)
print("number of interference =",len(h_idx))

#calculate the score
res= blk.calc_score(L_est_str)
print(res)