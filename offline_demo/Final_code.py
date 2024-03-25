#!/usr/bin/python3
import numpy as np
import sys
from sys import stdin
import Blackbox as blk
import os
import time

def safe_print(n):
    print(n)
    sys.stdout.flush() # Please flush the output for each print, otherwise it will result in a Time Limit Exceeded!

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

########################## Code starts ########################################

# ---Step 1: Input 'Start'---
line1 = stdin.readline().strip()

def test_large(vector, order=0.1):
    return np.sum(np.abs(vector[0])) > order

W2 = np.mat((np.random.randn(32,32) + 1j*np.random.randn(32,32)))
W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
W1[0,0] = 1.0 + 0.0*1j
W1 = bf_norm_multiple_stream(W1)
W2 = bf_norm_multiple_stream(W2)
input_01 = mtx2outputdata(W1)
input_02 = mtx2outputdata(W2)
rece_Y1 = blk.blackboxSystem(input_01, input_02)
Y = read_blackbox(rece_Y1)
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
        if N_tar != -1:
            if len(h_idx) >10:
                U, S, Vh = np.linalg.svd(Y01)
                tolerance = 1e-5 
                rank = np.sum(S > tolerance)
                N_tar = rank -1
            # assert rank == N_tar
            # assert np.linalg.matrix_rank(y_01) == N_tar
        else:
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
            U, S, Vh = np.linalg.svd(Y01)
            tolerance = 1e-5 
            rank = np.sum(S > tolerance)
            N_tar = rank    

#******************* Case 1 and case 2 work start athere ****************************

if len(h_idx) <= 10: # Case 1 and case 2 work here
    # 调参数 1：你可以 randomize clean_idx，你应还是能看得懂 clean_idx 代表着什么的
    # print(f"h_idx = {h_idx}")
    all_idx = set(range(32))
    for idx in h_idx:
        all_idx.discard(idx)
    remaining_idx = sorted(all_idx)
    clean_idx = remaining_idx[:N_tar]
    # print(f"clean_idx = {clean_idx}")
    
    H1_matrix_row_space_estimator = np.zeros((N_tar, 32)) + 1j * np.zeros((N_tar, 32))
    
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    
    # 调参数 2，3：
    # 这里先介绍什么参数可以调
    # 首先是你看我是有 3 个步骤的，先 compute a_square，再是 compute b_square，然后是 a_divide_sqrt_2_plus_b_divide_sqrt_2_square
    # 在每一个步骤里面，有 2 个参数可以调
    # 第一个是，W1 matrix，3 个步骤都是针对 column 0，你可以换成 column 1, 2, 3 (记住是 3 个步骤一起换)
    # 第二个是，Y_with_inference matrix 每次都只截取 column 0，你可以换成 0 to 299 任何一个数 (记住是 3 个步骤一起换)
    
    for i in range(32):
        if i==0:
            # Compute a_square
            W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            W1[0,0] = 1. # 可以换 column
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            a_square = Y_with_inference[clean_idx, 0] # 可以换 column
            # record a_square into the first entry
            H1_matrix_row_space_estimator[:,0] = a_square
        else:
            # Compute b_square
            W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            W1[i,0] = 1. # 可以换 column
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            b_square = Y_with_inference[clean_idx, 0]  # 可以换 column
            
            # Compute a_plus_b_square
            W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            W1[0, 0] = 1/np.sqrt(2) + 1j * 0 # 可以换 column
            W1[i, 0] = 1/np.sqrt(2) + 1j * 0 # 可以换 column
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            a_divide_sqrt_2_plus_b_divide_sqrt_2_square = Y_with_inference[clean_idx, 0] # 可以换 column
            
            # Record 
            ab = a_divide_sqrt_2_plus_b_divide_sqrt_2_square - ( a_square / 2 ) - ( b_square / 2 )
            H1_matrix_row_space_estimator[:, i] = ab
    
    L_est = np.mat(H1_matrix_row_space_estimator)
    L_est = norm_multiple_stream_result(L_est)
    safe_print('END')
    line2 = stdin.readline().strip()
    L_est = mtx2outputdata_result(L_est)
    Score = blk.calc_score(L_est)

#******************* Case 1 and case 2 work end at here ****************************




#******************* Case 3 and case 4 work start at here ****************************

else: # Case 3 and case 4 work here
    
    # Step 1: Prepare a ratio matrix
    
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    
    W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    W1[0, 0] = 1.
    W1 = bf_norm_multiple_stream(W1)
    
    input_weight_1 = mtx2outputdata(W1)
    input_weight_2 = mtx2outputdata(W2)
    
    Y_0 = blk.blackboxSystem(input_weight_1, input_weight_2)
    Y_0 = read_blackbox(Y_0)
    
    ratio_matrix = np.zeros((32, 300)) + 1j * np.zeros((32, 300))
    ratio_matrix[0,:] = 1.0 + 0.0 * 1j

    for i in range(1, 32):
        W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        W1[0,i] = 1.0 + 0.0*1j
        input_weight_1 = mtx2outputdata(W1)
        input_weight_2 = mtx2outputdata(W2)
        Y_i = blk.blackboxSystem(input_weight_1, input_weight_2)
        Y_i = read_blackbox(Y_i)
        ratio_matrix[i,:] = np.sqrt((Y_i / Y_0))[0,:]
    
    # Step 2: Create a giant rectangle 3-D box
    
    Y_giant = np.zeros((32, 300, 32)) + 1j * np.zeros((32, 300, 32))
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    for i in range(32):
        W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        W1[i, 0] = 1.
        W1 = bf_norm_multiple_stream(W1)
        input_weight_1 = mtx2outputdata(W1)
        input_weight_2 = mtx2outputdata(W2)
        Y_one_layer = blk.blackboxSystem(input_weight_1, input_weight_2)
        Y_one_layer = read_blackbox(Y_one_layer)
        Y_giant[:,:,i] = Y_one_layer
    
    # Step 3: Retrieve the h vectors
    
    # First, create a h_vectors_record
    h_vectors_record = np.zeros((32, 300)) + 1j * np.zeros((32, 300))
    
    # Then find the basis vectors for sheet 0 (trimmed first row, so sheet_0 dimension is 31 X 300)
    sheet_0 = Y_giant[1:,:,0]
    U, S, Vh = np.linalg.svd(sheet_0)
    basis_vectors = U[:, :N_tar]
    
    # Then need special algorithm to find the h vectors one by one
    for i in range(1, 32):
        sheet_i = Y_giant[:,:,i]
        if i==1:
            sheet_i_trimmed = sheet_i[2:,:]
            basis_vectors_trimmed = basis_vectors[1:,:]
        elif i==2:
            sheet_i_trimmed = np.concatenate((np.expand_dims(sheet_i[1,:],axis=0),sheet_i[3:,:]),axis=0)
            basis_vectors_trimmed = np.concatenate((np.expand_dims(basis_vectors[0,:],axis=0),basis_vectors[2:,:]),axis=0)
        elif i >= 3 and i <= 29:
            sheet_i_trimmed = np.concatenate((sheet_i[1:i,:],sheet_i[i+1:,:]),axis=0)
            basis_vectors_trimmed = np.concatenate((basis_vectors[0:i-1,:],basis_vectors[i:,:]),axis=0)
        elif i==30:
            sheet_i_trimmed = np.concatenate((sheet_i[1:30,:],np.expand_dims(sheet_i[31,:],axis=0)),axis=0)
            basis_vectors_trimmed = np.concatenate((basis_vectors[0:29,:],np.expand_dims(basis_vectors[30,:],axis=0)),axis=0)
        elif i==31:
            sheet_i_trimmed = sheet_i[1:31,:]
            basis_vectors_trimmed = basis_vectors[0:30,:]
        # print(sheet_i_trimmed.shape) # Will get (30,300)
        # print(basis_vectors_trimmed.shape) # Will get (30,10)
        
        # 去拿 10 X 300 的 coordinate matrix. 
        basis_vectors_pinv = np.linalg.pinv(basis_vectors_trimmed)
        Coordinate_Matrix = basis_vectors_pinv @ sheet_i_trimmed
        # print(Coordinate_Matrix.shape) # Will get (10,300)
        
        # 从 coordinate matrix 拿回原本的 31 X 300 matrix
        sheet_i_new = basis_vectors @ Coordinate_Matrix
        # print(sheet_i_new.shape) # Will get (31, 300)
        
        # 然后就可以找出 h vector 了
        h_vectors_record[i,:] = Y_giant[i,:,i] - sheet_i_new[i-1,:]
        
    # 也要找 sheet 1 的 h vector
    sheet_0 = Y_giant[:,:,0]
    sheet_1 = Y_giant[:,:,1]
    
    sheet_1_edit = np.concatenate((np.expand_dims(sheet_1[0,:],axis=0), sheet_1[2:,:]), axis=0) # (31 X 300)
    U, S, Vh = np.linalg.svd(sheet_1_edit)
    basis_vectors_of_sheet_1_edit = U[:, :N_tar] # (31 X 10)
    
    basis_vectors_of_sheet_1_edit_trimmed = basis_vectors_of_sheet_1_edit[1:,:] # (30 X 10)
    
    # 去拿 10 X 300 的 coordinate matrix. 
    basis_vectors_pinv = np.linalg.pinv(basis_vectors_of_sheet_1_edit_trimmed)
    Coordinate_Matrix = basis_vectors_pinv @ sheet_0[2:,:] # (10 X 300) 
    # The @ operator is eqiivalent to np.dot(A, B), meaning matrix multiply. 
    # Don't ask me why I came up with this. Ask ChatGPT 4.0
    # print(Coordinate_Matrix.shape) # Will get (10,300)

    # 从 coordinate matrix 拿回原本的 31 X 300 matrix
    sheet_0_new = np.dot(basis_vectors_of_sheet_1_edit, Coordinate_Matrix)
    # print(sheet_0_new.shape) # Will get (31, 300)

    # 然后就可以找出 h vector 了
    h_vectors_record[0,:] = Y_giant[0,:,0] - sheet_0_new[0,:]
    
    # 试试效果如何
    # print_numpy_array(h_vectors_record, "h_vectors_record")
    
    # Estimate real Y without inference
    def estimate_real_Y_without_inference(W1, Y): # W1 is (32 X 32)
        # print_numpy_array(np.asarray(W1), "W1")
        W1 = np.asarray(W1)
        W1 = np.conjugate(W1)
        A = np.dot(W1,ratio_matrix)
        A = np.multiply(A, A)
        return Y - np.multiply(A, h_vectors_record)
    
    # print("=============================")
    
    # 我们首先要解决一下 complex number 旋转的问题
    # 要回头去改我们的 ratio matrix
    
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    
    for i in range(1, 32):
        W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        W1[0, 0] = 1/np.sqrt(2) + 1j * 0
        W1[0, i] = 1/np.sqrt(2) + 1j * 0
        W1 = bf_norm_multiple_stream(W1)
    
        input_weight_1 = mtx2outputdata(W1)
        input_weight_2 = mtx2outputdata(W2)
        
        Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
        Y_with_inference = read_blackbox(Y_with_inference)
        
        estimated_real_Y_without_inference = estimate_real_Y_without_inference(W1, Y_with_inference)
        
        for j in range(300):
            if np.abs(estimated_real_Y_without_inference[0,j])>2: # 可以是 >1, >3 等等，总之看看那个值是否很离谱
                ratio_matrix[i,j] = -ratio_matrix[i,j]
    
    # 其实你会看得出来，我们估算的东西是很接近真实的数字的
    
    # print("=============================")
    
    H1_matrix_row_space_estimator = np.zeros((N_tar, 32)) + 1j * np.zeros((N_tar, 32))
    
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    
    # 调参数 4， 5：
    # 基本上重复上面的步骤
    # 首先是你看我是有 3 个步骤的，先 compute a_square，再是 compute b_square，然后是 a_divide_sqrt_2_plus_b_divide_sqrt_2_square
    # 在每一个步骤里面，有 2 个参数可以调
    # 第一个是，W1 matrix，3 个步骤都是针对 column 0，你可以换成 column 1, 2, 3 (记住是 3 个步骤一起换)
    # 第二个是，Y_with_inference matrix 每次都只截取 column 0，你可以换成 0 to 299 任何一个数 (记住是 3 个步骤一起换)
    
    # 调参数 6：
    # 然后 selected_idx 也可以调
    # 跟 clean_idx 同一个道理，但是在 case 3, case 4 基本上没有什么 clean 不 clean index 的了，所有 index 都是被污染过的
    # 所以叫做 selected_idx，没有 restriction 随便选
    selected_idx = [i for i in range(1, N_tar+1)]
    
    for i in range(32):
        if i==0:
            # Compute a_square
            W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            W1[0,0] = 1. # 可以换 column
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            estimated_real_Y_without_inference = np.asarray(estimate_real_Y_without_inference(W1, Y_with_inference))
            a_square = estimated_real_Y_without_inference[selected_idx, 0] # 可以换 column
            # record a_square into the first entry
            H1_matrix_row_space_estimator[:,0] = a_square
        else:
            # Compute b_square
            W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            W1[i,0] = 1. # 可以换 column
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            estimated_real_Y_without_inference = np.asarray(estimate_real_Y_without_inference(W1, Y_with_inference))
            b_square = estimated_real_Y_without_inference[selected_idx, 0] # 可以换 column
            # Compute a_plus_b_square
            W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            W1[0, 0] = 1/np.sqrt(2) + 1j * 0 # 可以换 column
            W1[i, 0] = 1/np.sqrt(2) + 1j * 0 # 可以换 column
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            estimated_real_Y_without_inference = np.asarray(estimate_real_Y_without_inference(W1, Y_with_inference))
            a_plus_b_square = estimated_real_Y_without_inference[selected_idx, 0] # 可以换 column
            # Record 
            ab = a_plus_b_square - ( a_square / 2 ) - ( b_square / 2 )
            H1_matrix_row_space_estimator[:, i] = ab
    
    L_est = np.mat(H1_matrix_row_space_estimator)
    L_est = norm_multiple_stream_result(L_est)
    safe_print('END')
    line2 = stdin.readline().strip()
    L_est = mtx2outputdata_result(L_est)
    Score = blk.calc_score(L_est)

#******************* Case 3 and case 4 work end at here ****************************

