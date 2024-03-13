#!/usr/bin/python3
import numpy as np
import sys
from sys import stdin



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
    safe_print(output)
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
    safe_print(output)
    return output

def read_blackbox():
    line = stdin.readline().strip()
    m = line.split(' ')
    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        m[ii] = float(m[ii])
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j
    n = n.reshape(300,32) # This step is to reshape Y to a matrix
    n = n.T # This step is to match the size of Y in the document
    return n

N_stream1 = 32
def find_inter() -> list:
    '''ratio_matrix is expected to be 32,300 shape
       y is expected to be 32,300 shape
       w1 is expected to be 32,32 shape, with first column have one 1, the other all 0
       w2 is expected to be 32,32 shape, and we will fixed w2 elements to 1/32'''

    indices_and_norms = []
    inter_list = []
    for i in range(32):
        W1 = np.mat((np.zeros((32,N_stream1)) + 1j*np.zeros((32,N_stream1))))
        W1[i,0] = 1.0 + 0.0j
        W2 = W1
        # W1 W2 Normalization
        # Output W1 W2
        mtx2outputdata(W1)
        mtx2outputdata(W2)
        # Input Y
        Y = read_blackbox()

        magnitude = np.linalg.norm(Y[i, :]) 
        indices_and_norms.append([i, magnitude]) 
    
    for pair in indices_and_norms:
        if pair[1] >500:
            inter_list.append(pair[0])
    return inter_list



# ---Step 1: Input 'Start'---
line1 = stdin.readline().strip()

# ---Step 1: Output W1 W2---
# N_stream1, N_stream2 can be [1,32]
N_stream1 = 32
N_stream2 = 32
# W1, W2 Example
W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
W2 = np.mat((np.random.randn(32,N_stream2) + 1j*np.random.randn(32,N_stream2)))
# W1 W2 Normalization
W1 = bf_norm_multiple_stream(W1)
W2 = bf_norm_multiple_stream(W2)
# Output W1 W2
mtx2outputdata(W1)
mtx2outputdata(W2)

# ---Step 3: Input Y---
Y = read_blackbox()

# ---Step 3: Output another W1 W2 for each interaction process (if interaction continue)---
inter_num = 10 #Contestants can control the number of interaction
# N_stream1, N_stream2 can be [1,...,32]
N_stream1 = 32
N_stream2 = 32
# W1, W2 Example
W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
W2 = np.mat((np.random.randn(32,N_stream2) + 1j*np.random.randn(32,N_stream2)))
for ii in range(inter_num-1):
    # W1 W2 Normalization
    W1 = bf_norm_multiple_stream(W1)
    W2 = bf_norm_multiple_stream(W2)
    # Output W1 W2
    mtx2outputdata(W1)
    mtx2outputdata(W2)
    # Input Y
    Y = read_blackbox()

    # Contestants can write their code here to process the Y data to obtain the W1 and W2 they want to input for the next iteration.

    #******Example code*******
    N_stream1 = 32
    N_stream2 = 32
    W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
    W2 = np.mat((np.random.randn(32,N_stream2) + 1j*np.random.randn(32,N_stream2)))
    #******Example code end*******


#At the end of interation, the estimated L_est need to be obtained.

#******Example code*******
# N_target can be [1,10]

    
N_infr = len(find_inter(Y))
N_target = len(Y[0])-N_infr

L_est = np.mat((np.random.randn(N_target,32) + 1j*np.random.randn(N_target,32)))
# L_est Normalization
L_est = norm_multiple_stream_result(L_est)
#******Example code end*******


# ---Step 3: Output 'END' (if interaction end)---
safe_print('END')

# ---Step 5: Input 'Roger that'---
line2 = stdin.readline().strip()

# ---Step 5: Output L_est---

# Output L_est
mtx2outputdata_result(L_est)
