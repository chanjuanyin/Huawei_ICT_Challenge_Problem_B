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
    input_data_ravel = input_data.ravel(order="F").reshape(1,-1) # matrix to vector
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

W1 = np.zeros((32, 32)) + 1j * np.zeros((32, 32))
W2 = (np.ones((32, 32)) + 1j * np.zeros((32, 32))) * (1/32)
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

W1 = np.zeros((32, 32)) + 1j * np.zeros((32, 32))
W2 = (np.ones((32, 32)) + 1j * np.zeros((32, 32))) * (1/32)
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



# ---Step 1: Input 'Start'---
#line1 = stdin.readline().strip() #For the submission code, we need this line to read the 'Start'

# ---Step 1: Output W1 W2---
# N_stream1, N_stream2 can be [1,32]
N_stream1 = 4 
N_stream2 = 4
# W1, W2 Example
W1 = np.mat(np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1))
W2 = np.mat(np.random.randn(32,N_stream2) + 1j*np.random.randn(32,N_stream2))
# W1 W2 Normalization
W1 = bf_norm_multiple_stream(W1)
W2 = bf_norm_multiple_stream(W2)
print_numpy_array(W1, "W1")
print_numpy_array(W2, "W2")
# Output W1 W2
input_weight_1 = mtx2outputdata(W1)
input_weight_2 = mtx2outputdata(W2)

rece_Y = blk.blackboxSystem(input_weight_1, input_weight_2)
# ---Step 3: Input Y---
Y = read_blackbox(rece_Y)
print_numpy_array(Y, "Y")

# ---Step 3: Output another W1 W2 for each interaction process (if interaction continue)---
inter_num = 100 #Contestants can control the number of interaction
# N_stream1, N_stream2 can be [1,...,32]
N_stream1 = 1 
N_stream2 = 32
# W1, W2 Example
W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
W2 = np.mat((np.random.randn(32,N_stream2) + 1j*np.random.randn(32,N_stream2)))
for ii in range(inter_num-1):
    # W1 W2 Normalization
    W1 = bf_norm_multiple_stream(W1)
    W2 = bf_norm_multiple_stream(W2)
    # Output W1 W2
    input_weight_1 = mtx2outputdata(W1)
    input_weight_2 = mtx2outputdata(W2)

    rece_Y = blk.blackboxSystem(input_weight_1, input_weight_2)
    # Input Y
    Y = read_blackbox(rece_Y)

    # Contestants can write their code here to process the Y data to obtain the W1 and W2 they want to input for the next iteration.

    #******Example code*******
    N_stream1 = 1 
    N_stream2 = 32
    W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
    W2 = np.mat((np.random.randn(32,N_stream2) + 1j*np.random.randn(32,N_stream2)))
    #******Example code end*******
    

#At the end of interation, the estimated L_est need to be obtained.

#******Example code*******
# N_target can be [1,10]
N_target = 4 # Notices that the N_target is different for each case. Contestants program need to estimate the number of target signals first.

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
