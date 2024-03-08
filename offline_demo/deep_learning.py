#!/usr/bin/python3
import numpy as np
import sys
from sys import stdin
import Blackbox as blk
import os
import time


def safe_print(n):
    print(n)
    sys.stdout.flush()

def print_numpy_array(complex_array, file_name = "np_array"):
    directory_path = os.path.join("offline_demo", "print_directory")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    file_path = os.path.join(directory_path, file_name + ".csv")
        
    dimensions = complex_array.ndim
    shape = complex_array.shape
    # Open a CSV file for writing
    with open(file_path, 'w') as file:
        # Function to flatten the array and iterate through its elements
        def iterate_and_print(array):
            for element in np.nditer(array):
                real_part = element.real
                imag_part = element.imag
                file.write(f"{real_part} {imag_part}, ")

        # Handle different dimensions
        if dimensions == 1:
            iterate_and_print(complex_array)
        elif dimensions == 2:
            for row in range(shape[0]):
                iterate_and_print(complex_array[row, :])
                file.write("\n")
        elif dimensions == 3:
            for depth in range(shape[0]):
                file.write(f"Depth {depth + 1}:\n")
                for row in range(shape[1]):
                    iterate_and_print(complex_array[depth, row, :])
                    file.write("\n")
                file.write("\n")
        else:
            print("Unsupported dimension for CSV export.")


def bf_norm_multiple_stream(bf_new):
    bf_norm = bf_new/np.linalg.norm(bf_new)
    return bf_norm

def norm_multiple_stream_result(L_est):
    stream_num = L_est.shape[0]
    for ii in range(stream_num):
        L_est[ii,:] = L_est[ii,:]/np.linalg.norm(L_est[ii,:])
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
    #safe_print(output)
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
    #safe_print(output)
    return output

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



# Try here:

W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
W1[0,0] = 1.
W1 = bf_norm_multiple_stream(W1)
W2 = bf_norm_multiple_stream(W2)
input_weight_1 = mtx2outputdata(W1)
input_weight_2 = mtx2outputdata(W2)
start_time = time.time()
rece_Y1 = blk.blackboxSystem(input_weight_1, input_weight_2)
Y1 = read_blackbox(rece_Y1)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print(Y1.shape)

W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
W1[0,1] = 1.
W1 = bf_norm_multiple_stream(W1)
W2 = bf_norm_multiple_stream(W2)
input_weight_1 = mtx2outputdata(W1)
input_weight_2 = mtx2outputdata(W2)
start_time = time.time()
rece_Y2 = blk.blackboxSystem(input_weight_1, input_weight_2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
Y2 = read_blackbox(rece_Y2)
print(Y2.shape)

Y_diff = Y1 / Y2
print_numpy_array(Y_diff, "Y_diff")


#*************** trying deep learning ********************


Y_giant = np.zeros((32, 300, 32)) + 1j * np.zeros((32, 300, 32))
W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
W2 = bf_norm_multiple_stream(W2)
input_weight_2 = mtx2outputdata(W2)
# start_time = time.time()
for i in range(32):
    W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    W1[i, 0] = 1.
    W1 = bf_norm_multiple_stream(W1)
    input_weight_1 = mtx2outputdata(W1)
    Y_one_layer = blk.blackboxSystem(input_weight_1, input_weight_2)
    Y_one_layer = read_blackbox(Y_one_layer)
    Y_giant[:,:,i] = Y_one_layer
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")
# print(Y_giant.shape)
# print_numpy_array(Y_giant[:,:,5], "Y_giant_6")

global Y_matrix, N_target

for i in range(32):
    for j in range(0, i):
        Y_matrix = np.concatenate((Y_matrix, np.expand_dims(Y_giant[j,:,i], axis=0)), axis=0)
    for j in range(i+1, 32):
        if i==0 and j==1:
            Y_matrix = np.expand_dims(Y_giant[j,:,i], axis=0)
        else:
            Y_matrix = np.concatenate((Y_matrix, np.expand_dims(Y_giant[j,:,i], axis=0)), axis=0)
# print(Y_matrix.shape)
Y_matrix = np.mat(Y_matrix)

N_target = 6

# First N_target rows are H^2
# Next N_target rows are L^2
X_initializer = np.random.rand(N_target*2, 32) - 0.5 + 1j * (np.random.rand(N_target*2, 32) - 0.5)
X_initializer = (X_initializer.T / np.linalg.norm(X_initializer, axis = 1)).T

def compute_loss(X_initializer):

    for i in range(N_target):
        sheet = np.outer(X_initializer[i, :], X_initializer[i+N_target, :])
        # print_numpy_array(sheet, "sheet_{}".format(i))
        for j in range(32):
            if j==0:
                X_vector = sheet[1:,0]
            else:
                X_vector = np.concatenate((X_vector, sheet[0:j, j]))
                X_vector = np.concatenate((X_vector, sheet[j+1:32, j]))
        # print(X_vector.shape)
        if i==0:
            X_matrix = np.expand_dims(X_vector, axis=0)
        else:
            X_matrix = np.concatenate((X_matrix, np.expand_dims(X_vector, axis=0)), axis=0)
    X_matrix = X_matrix.T
    # print(X_matrix.shape)

    I = np.eye(992, dtype=np.complex128)
    X_hermitian = np.conjugate(X_matrix.T)
    XX_hermitian_inv = np.linalg.inv(np.dot(X_hermitian, X_matrix))
    intermediate_matrix = I - np.dot(np.dot(X_matrix, XX_hermitian_inv), X_hermitian)
    error_matrix = np.dot(intermediate_matrix, Y_matrix)
    loss = np.linalg.norm(error_matrix, 'fro')
    # print(error_matrix.shape)
    return loss

start_time = time.time()
loss = compute_loss(X_initializer)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print(loss)



#*************** end of trying deep learning ********************

#******Example code*******
# N_target can be [1,10]
N_target = 6 # Notices that the N_target is different for each case. Contestants program need to estimate the number of target signals first.

L_est = np.mat((np.random.randn(N_target,32) + 1j*np.random.randn(N_target,32)))
# L_est Normalization
L_est = norm_multiple_stream_result(L_est)
#******Example code end*******

    
# ---Step 3: Output 'END' (when interaction end)---
#safe_print('END') #For the submission code, we need this line to tell blackbox to calc the score

# ---Step 5: Input 'Roger that'---
#line2 = stdin.readline().strip()#For the submission code, we need this line to read the 'Roger that'

# ---Step 5: Output L_est---

# Output L_est
L_est = mtx2outputdata_result(L_est)
Score = blk.calc_score(L_est)
