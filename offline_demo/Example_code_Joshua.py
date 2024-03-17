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
        print('haha')
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

#******************* Case 1 and case 2 work start athere ****************************

if len(h_idx) <= 10: # Case 1 and case 2 work here
    pass

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
        print_numpy_array(np.sqrt(Y_i/Y_0), f"Y_{i}_divides_Y_0")
        ratio_matrix[i,:] = np.sqrt((Y_i / Y_0))[0,:]
    
    print(ratio_matrix.shape)
    print_numpy_array(ratio_matrix, "ratio_matrix")
    
    # Step 2: Create a giant rectangle 3-D box
    
    Y_giant = np.zeros((32, 300, 32)) + 1j * np.zeros((32, 300, 32))
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    # start_time = time.time()
    for i in range(32):
        W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        W1[i, 0] = 1.
        W1 = bf_norm_multiple_stream(W1)
        input_weight_1 = mtx2outputdata(W1)
        input_weight_2 = mtx2outputdata(W2)
        Y_one_layer = blk.blackboxSystem(input_weight_1, input_weight_2)
        Y_one_layer = read_blackbox(Y_one_layer)
        Y_giant[:,:,i] = Y_one_layer
    
    print_numpy_array(Y_giant[:,:,0], "sheet_0_original")
    print_numpy_array(Y_giant[:,:,1], "sheet_1_original")
    print_numpy_array(Y_giant[:,:,2], "sheet_2_original")
    print_numpy_array(Y_giant[:,:,3], "sheet_3_original")
    
    # Step 3: Retrieve the h vectors
    
    # First, create a h_vectors_record
    h_vectors_record = np.zeros((32, 300)) + 1j * np.zeros((32, 300))
    
    # Then find the basis vectors for sheet 0 (trimmed first row, so sheet_0 dimension is 31 X 300)
    sheet_0 = Y_giant[1:,:,0]
    U, S, Vh = np.linalg.svd(sheet_0)
    basis_vectors = U[:, :N_tar]
    
    # Then need special algorithm to find the h vectors one by one
    for i in range(1, 32):
        # print(f"i = {i}")
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
    print_numpy_array(h_vectors_record, "h_vectors_record")
    
    # Estimate real Y without inference
    def estimate_real_Y_without_inference(W1, Y): # W1 is (32 X 32)
        print_numpy_array(np.asarray(W1), "W1")
        W1 = np.asarray(W1)
        W1 = np.conjugate(W1)
        A = np.dot(W1,ratio_matrix)
        A = np.multiply(A, A)
        return Y - np.multiply(A, h_vectors_record)
    
    print("=============================")
    
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
    
    print("=============================")
    
    # Assume 解决了
    
    W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
    W1 = bf_norm_multiple_stream(W1)
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    
    input_weight_1 = mtx2outputdata(W1)
    input_weight_2 = mtx2outputdata(W2)
    
    Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
    Y_with_inference = read_blackbox(Y_with_inference)
    print_numpy_array(Y_with_inference, "Y_with_inference")
    
    estimated_real_Y_without_inference = estimate_real_Y_without_inference(W1, Y_with_inference)
    print_numpy_array(estimated_real_Y_without_inference, "estimated_real_Y_without_inference")
    
    real_Y_without_inference = blk.blackboxSystem_no_h(input_weight_1, input_weight_2)
    real_Y_without_inference = read_blackbox(real_Y_without_inference)
    print_numpy_array(real_Y_without_inference, "real_Y_without_inference")
    
    # 其实你会看得出来，我们估算的东西是很接近真实的数字的
    
    # print("=============================")
    
    # # 这一段我们可以忽略，是我在 debug 过程的一部分
    
    # W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    # W1[0, 0] = 1/np.sqrt(2) + 1j * 0
    # W1[0, 1] = 1/np.sqrt(2) + 1j * 0
    # input_weight_1 = mtx2outputdata(W1)
    # input_weight_2 = mtx2outputdata(W2)
    # Y_mixed = blk.blackboxSystem(input_weight_1, input_weight_2)
    # Y_mixed = read_blackbox(Y_mixed)
    # print_numpy_array(Y_mixed/Y_0, f"Y_mixed_divides_Y_0")
    
    # W1 = np.asarray(W1)
    # W1 = np.conjugate(W1)
    # A = np.dot(W1,ratio_matrix)
    # # A = np.multiply(A, A)
    # print_numpy_array(A, f"A_matrix_test")
    
    print("=============================")
    
    print("跑跑实验 1")
    
    # 跑跑实验
    
    H1_matrix = blk.return_H1_matrix()
    print(H1_matrix.shape)
    print_numpy_array(H1_matrix, "H1_matrix")
    L_est = np.mat(H1_matrix)
    L_est = norm_multiple_stream_result(L_est)
    safe_print('END')
    line2 = stdin.readline().strip()
    L_est = mtx2outputdata_result(L_est)
    Score = blk.calc_score(L_est)
    print(f"Score = {Score}")
    
    print("=============================")
    
    print("跑跑实验 2")
    
    # 跑跑实验 2
    
    H1_matrix = blk.return_H1_matrix()
    print(H1_matrix.shape)
    print_numpy_array(H1_matrix, "H1_matrix")
    U, S, Vh = np.linalg.svd(H1_matrix)
    tolerance = 1e-10
    null_mask = (S <= tolerance)
    null_indices = np.where(null_mask)[0]
    null_space_vectors = Vh[-(32-N_tar):, :]
    print(f"null_space_vectors.shape = {null_space_vectors.shape}")
    U, S, Vh = np.linalg.svd(null_space_vectors)
    tolerance = 1e-10
    null_mask = (S <= tolerance)
    null_indices = np.where(null_mask)[0]
    L_est = Vh[-(N_tar):, :]
    print(f"L_est.shape = {L_est.shape}")
    L_est = np.mat(L_est)
    L_est = norm_multiple_stream_result(L_est)
    safe_print('END')
    line2 = stdin.readline().strip()
    L_est = mtx2outputdata_result(L_est)
    Score = blk.calc_score(L_est)
    print(f"Score = {Score}")
    
    print("=============================")
    
    print("跑跑实验 3")
    
    # 继续哦
    # 解释一下我在干什么，这一步的目的是要算出 H1_matrix_square_mixed
    # 而 H1_matrix_square_mixed 的 row vector 的 linear span 基本上代表了 np.square(H1_matrix) 的 row vectors 的 linear span
    # 注意 np.square(H1_matrix) 的意思是 H1 matrix 去 element-wise square
    # 拿到了 H1_matrix_square_mixed，基本上已经很接近很接近我们需要的答案了
    # 可是再进一步我目前还没有想到该如何 proceed
    
    H1_matrix_square_mixed = np.zeros((N_tar, 32)) + 1j * np.zeros((N_tar, 32))
    
    W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    W2 = bf_norm_multiple_stream(W2)
    for j in range(N_tar):
        for i in range(32):
            v = ratio_matrix[:,j]
            v = v.reshape((32,1))
            # x = np.zeros((32, 1)) + 1j * np.zeros((32, 1))
            # x[i][0] =  1.0 + 0.0 * 1j
            A = np.zeros((32, 32), dtype=complex)
            A[i][i] = np.conjugate(1 / v[i][0])
            # print_numpy_array(np.dot(np.conjugate(A), v), f"A_mul_v_at_row_{i}")
            W1 = np.mat(A)
            W1 = bf_norm_multiple_stream(W1)
            input_weight_1 = mtx2outputdata(W1)
            input_weight_2 = mtx2outputdata(W2)
            Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
            Y_with_inference = read_blackbox(Y_with_inference)
            estimated_real_Y_without_inference = np.asarray(estimate_real_Y_without_inference(W1, Y_with_inference))
            H1_matrix_square_mixed[j, i] = estimated_real_Y_without_inference[0,j]
    
    # H1_matrix_square_mixed = (H1_matrix_square_mixed.T / np.linalg.norm(H1_matrix_square_mixed, axis = 1)).T
    # print(f"np.square(H1_matrix).shape = {np.square(H1_matrix).shape}")
    # print(f"H1_matrix_square_mixed.shape = {H1_matrix_square_mixed.shape}")
    print_numpy_array(np.square(H1_matrix), "np.square(H1_matrix)")
    print_numpy_array(H1_matrix_square_mixed, "H1_matrix_square_mixed")
    
    # 验证是否正确
    # 先要去拿 coordinate matrix
    H1_matrix_square_pinv = np.linalg.pinv(np.square(H1_matrix).T)
    Coordinate_Matrix = H1_matrix_square_pinv @ H1_matrix_square_mixed.T
    # 从 coordinate matrix 拿回原本的 H1_matrix_square_mixed matrix
    H1_matrix_square_mixed_new = np.square(H1_matrix).T @ Coordinate_Matrix
    H1_matrix_square_mixed_new = H1_matrix_square_mixed_new.T
    print_numpy_array(H1_matrix_square_mixed_new, f"H1_matrix_square_mixed_new")
    
    # 验证出来我的想法是成功的
    
    # 可是依然不是最终的结果
    L_est = np.mat(H1_matrix_square_mixed)
    L_est = norm_multiple_stream_result(L_est)
    safe_print('END')
    line2 = stdin.readline().strip()
    L_est = mtx2outputdata_result(L_est)
    Score = blk.calc_score(L_est)
    print(f"Score = {Score}")
    
    
    print("=============================")
    
    # print("跑跑实验 4")
    
    # # 继续哦
    
    # giant_sort_array = []
    # W1_record = []
    
    # W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    # W2 = bf_norm_multiple_stream(W2)
    
    # number_of_trials = 1000
    # print(f"number_of_trials = {number_of_trials}")
    # for i in range(number_of_trials): # 跑 1000 次也好不了多少
    #     W1 = np.mat((np.random.randn(32,N_stream1) + 1j*np.random.randn(32,N_stream1)))
    #     W1 = bf_norm_multiple_stream(W1)
    #     W1_record.append(W1)
    #     input_weight_1 = mtx2outputdata(W1)
    #     input_weight_2 = mtx2outputdata(W2)
    #     Y_with_inference = blk.blackboxSystem(input_weight_1, input_weight_2)
    #     Y_with_inference = read_blackbox(Y_with_inference)
    #     estimated_real_Y_without_inference = estimate_real_Y_without_inference(W1, Y_with_inference)
    #     norms = np.linalg.norm(estimated_real_Y_without_inference, axis=0)
    #     for j in range(300):
    #         giant_sort_array.append(tuple([norms[j], i, j]))
    
    # giant_sort_array = sorted(giant_sort_array, key=lambda x: x[0])
    
    # null_space_vectors_main = np.zeros((32, 32-N_tar)) + 1j * np.zeros((32, 32-N_tar))
    # null_space_vectors_check_rank = np.zeros((32, 32-N_tar)) + 1j * np.zeros((32, 32-N_tar))
    # count = 0
    # k = 0
    # while count < 32-N_tar and k < 200:
    #     print(f"At position k = {k} we have : {giant_sort_array[k]}")
    #     W1 = W1_record[giant_sort_array[k][1]]
    #     W1 = np.asarray(W1)
    #     W1 = np.conjugate(W1)
    #     v1 = ratio_matrix[:,giant_sort_array[k][2]]
    #     v1 = v1.reshape((32, 1))
    #     v2 = np.dot(W1, v1)
    #     print_numpy_array(np.dot(H1_matrix, v1), f"check_if_null_enough_{k}")
    #     null_space_vectors_check_rank[:,count] = v2[:,0]
    #     U, S, Vh = np.linalg.svd(null_space_vectors_check_rank)
    #     tolerance = 0.1 # need a more lenient tolerance, such that we can reject a not so linearly independent vector
    #     rank = np.sum(S > tolerance)
    #     print(f"rank = {rank}")
    #     if rank == count+1:
    #         print("accepted")
    #         null_space_vectors_main[:,count] = v2[:,0]
    #         count += 1
    #     else:
    #         print("rejected")
    #     k += 1
    
    # # for k in range(32-N_tar):
    # #     print(f"At position k we have : {giant_sort_array[k]}")
    # #     W1 = W1_record[giant_sort_array[k][1]]
    # #     W1 = np.asarray(W1)
    # #     W1 = np.conjugate(W1)
    # #     v1 = ratio_matrix[:,giant_sort_array[k][2]]
    # #     v1 = v1.reshape((32, 1))
    # #     v2 = np.dot(W1, v1)
    # #     null_space_vectors[:,k] = v2[:,0]
    
    # null_space_vectors = null_space_vectors_main.reshape((32-N_tar, 32))
    # null_space_vectors = np.mat(null_space_vectors)
    # null_space_vectors = norm_multiple_stream_result(null_space_vectors)
    # null_space_vectors = np.asarray(null_space_vectors)
    # print_numpy_array(null_space_vectors, "null_space_vectors")
    
    # U, S, Vh = np.linalg.svd(null_space_vectors)
    # tolerance = 1e-10
    # null_mask = (S <= tolerance)
    # null_indices = np.where(null_mask)[0]
    # L_est = Vh[-(N_tar):, :]
    # print(f"L_est.shape = {L_est.shape}")
    # L_est = np.mat(L_est)
    # L_est = norm_multiple_stream_result(L_est)
    # safe_print('END')
    # line2 = stdin.readline().strip()
    # L_est = mtx2outputdata_result(L_est)
    # Score = blk.calc_score(L_est)
    # print(f"Score = {Score}")
    
    # print("=============================")
    
    # print("向 deep learning 发起最后一次冲锋")
    
    # for i in range(32):
    #     Y_giant[i,:,i] = Y_giant[i,:,i] - h_vectors_record[i,:]
    
    # sheet_0 = Y_giant[:,:,1]
    # U, S, Vh = np.linalg.svd(sheet_0)
    # basis_vectors = U[:, :N_tar]
    
    # for i in range(32):
    #     if i==0:
    #         Y_matrix = Y_giant[:,:,0]
    #     else:
    #         Y_matrix = np.concatenate((Y_matrix, Y_giant[:,:,i]), axis=0)
    # Y_matrix = np.mat(Y_matrix)
    # print(f"Y_matrix.shape = {Y_matrix.shape}")
    
    # # H1_initializer are referring to H1 matrix
    # # basis_vectors refer to H3 matrix
    # H1_initializer = np.random.rand(N_tar, 32) - 0.5 + 1j * (np.random.rand(N_tar, 32) - 0.5)
    # H1_initializer = (H1_initializer.T / np.linalg.norm(H1_initializer, axis = 1)).T
    # basis_vectors = basis_vectors.T

    # def compute_loss(H1_initializer):

    #     for i in range(N_tar):
    #         sheet = np.outer(basis_vectors[i, :], H1_initializer[i, :])
    #         # print_numpy_array(sheet, "sheet_{}".format(i))
    #         X_vector = sheet.T.reshape((32 * 32,1))
    #         if i==0:
    #             X_matrix = X_vector
    #         else:
    #             X_matrix = np.concatenate((X_matrix, X_vector), axis=1)
    #     # print(f"X_matrix.shape = {X_matrix.shape}")

    #     I = np.eye(1024, dtype=np.complex128)
    #     X_hermitian = np.conjugate(X_matrix.T)
    #     XX_hermitian_inv = np.linalg.inv(np.dot(X_hermitian, X_matrix))
    #     intermediate_matrix = I - np.dot(np.dot(X_matrix, XX_hermitian_inv), X_hermitian)
    #     error_matrix = np.dot(intermediate_matrix, Y_matrix)
    #     loss = np.linalg.norm(error_matrix, 'fro')
    #     # print(error_matrix.shape)
    #     return loss

    # start_time = time.time()
    # loss = compute_loss(H1_initializer)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # # safe_print(f"Elapsed time: {elapsed_time} seconds")
    # # safe_print(loss)

    # epoch_total = 25
    # finite_difference = 0.0001
    # learning_rate = []
    # learning_rate_helper = [1, 1/2, 1/4, 1/8, 1/16] * 2
    # for rate in learning_rate_helper:
    #     for j in range(5):
    #         learning_rate.append(rate)

    # def adam_optimizer(H1_initializer, gradients, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    #     # Update biased first moment estimate
    #     m = beta1 * m + (1 - beta1) * gradients

    #     # Update biased second raw moment estimate
    #     v = beta2 * v + (1 - beta2) * np.square(gradients)

    #     # Compute bias-corrected first moment estimate
    #     m_hat = m / (1 - beta1 ** t)

    #     # Compute bias-corrected second raw moment estimate
    #     v_hat = v / (1 - beta2 ** t)

    #     # Update the weights
    #     H1_initializer -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    #     return H1_initializer, m, v

    # for epoch_num in range(epoch_total+1):
    #     start_time = time.time()
    #     print_numpy_array(H1_initializer, f"H1_initializer_epoch_{epoch_num}")
    #     loss_this_epoch = compute_loss(H1_initializer)
    #     L_est = np.sqrt(H1_initializer)
    #     L_est = (L_est.T / np.linalg.norm(L_est, axis = 1)).T
    #     L_est = norm_multiple_stream_result(np.mat(L_est))
    #     L_est = mtx2outputdata_result(L_est)
    #     Score = blk.calc_score(L_est)
    #     safe_print(f"Epoch number = {epoch_num} ; loss = {loss_this_epoch} ; final score = {Score}")
    #     if epoch_num == epoch_total:
    #         break
    #     gradient = np.zeros((N_tar, 32)) + 1j * np.zeros((N_tar, 32))
    #     for i in range(N_tar):
    #         for j in range(32):
    #             temp1 = np.zeros((N_tar, 32)) + 1j * np.zeros((N_tar, 32))
    #             temp1[i,j] = finite_difference * 1.
    #             new_loss = compute_loss(H1_initializer + temp1)
    #             gradient_val = (new_loss - loss_this_epoch) / finite_difference * 1.
    #             temp2 = np.zeros((N_tar, 32)) + 1j * np.zeros((N_tar, 32))
    #             temp2[i,j] = finite_difference * 1j
    #             new_loss = compute_loss(H1_initializer + temp2)
    #             gradient_val += (new_loss - loss_this_epoch) / finite_difference * 1j
    #             gradient[i,j] = gradient_val
    #     print_numpy_array(gradient, f"gradient_epoch_{epoch_num}")
        
    #     # ***************** Normal Optimizer *********************
    #     H1_initializer = H1_initializer - gradient * learning_rate[epoch_num]
    #     # ***************** End of Normal Optimizer *********************
        
    #     # # ***************** Adam Optimizer *********************
    #     # if epoch_num==0:
    #     #     m = np.zeros_like(H1_initializer)
    #     #     v = np.zeros_like(H1_initializer)
    #     # H1_initializer, m, v = adam_optimizer(H1_initializer, gradient, m, v, epoch_num+1, learning_rate[epoch_num])
    #     # # ***************** End of Adam Optimizer *********************
        
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     safe_print(f"Elapsed time: {elapsed_time} seconds")

    # #*************** end of trying deep learning ********************
    
    # print("=============================")
    
    # print("模仿 deep learning 的手法")
    
    # for i in range(32):
    #     Y_giant[i,:,i] = Y_giant[i,:,i] - h_vectors_record[i,:]
    
    # # basis_vectors refer to H3 matrix
    # sheet_0 = Y_giant[:,:,1]
    # U, S, Vh = np.linalg.svd(sheet_0)
    # basis_vectors = U[:, :N_tar]
    # basis_vectors = basis_vectors.T # basis_vectors refer to H3 matrix
    # H3_vectors = (basis_vectors.T / np.linalg.norm(basis_vectors, axis = 1)).T
    
    # for i in range(32):
    #     if i==0:
    #         Y_matrix = Y_giant[:,:,0]
    #     else:
    #         Y_matrix = np.concatenate((Y_matrix, Y_giant[:,:,i]), axis=0)
    # Y_matrix = np.mat(Y_matrix)
    # print(f"Y_matrix.shape = {Y_matrix.shape}")

    # def compute_loss(H1_initializer_square):

    #     for i in range(N_tar):
    #         sheet = np.outer(H3_vectors[i, :], H1_initializer_square[i, :])
    #         # print_numpy_array(sheet, "sheet_{}".format(i))
    #         X_vector = sheet.T.reshape((32 * 32,1))
    #         if i==0:
    #             X_matrix = X_vector
    #         else:
    #             X_matrix = np.concatenate((X_matrix, X_vector), axis=1)
    #     # print(f"X_matrix.shape = {X_matrix.shape}")

    #     I = np.eye(1024, dtype=np.complex128)
    #     X_hermitian = np.conjugate(X_matrix.T)
    #     XX_hermitian_inv = np.linalg.inv(np.dot(X_hermitian, X_matrix))
    #     intermediate_matrix = I - np.dot(np.dot(X_matrix, XX_hermitian_inv), X_hermitian)
    #     error_matrix = np.dot(intermediate_matrix, Y_matrix)
    #     loss = np.linalg.norm(error_matrix, 'fro')
    #     # print(error_matrix.shape)
    #     return loss

    # # H1_initializer are referring to H1 matrix
    # # basis_vectors refer to H3 matrix
    # start_time = time.time()
    # H1_initializer = np.random.rand(N_tar, 32) - 0.5 + 1j * (np.random.rand(N_tar, 32) - 0.5)
    # H1_initializer = (H1_initializer.T / np.linalg.norm(H1_initializer, axis = 1)).T
    # loss = compute_loss(np.square(H1_initializer))
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # safe_print(f"Elapsed time: {elapsed_time} seconds")
    # safe_print(f"loss = {loss}\n")
    
    # H1_initializer_list = []
    # record_list = []
    # number_of_random = 10000
    # start_time = time.time()
    # for i in range(number_of_random):
    #     # H1_initializer are referring to H1 matrix
    #     # basis_vectors refer to H3 matrix
    #     H1_initializer = np.random.rand(N_tar, 32) - 0.5 + 1j * (np.random.rand(N_tar, 32) - 0.5)
    #     H1_initializer = (H1_initializer.T / np.linalg.norm(H1_initializer, axis = 1)).T
    #     loss = compute_loss(np.square(H1_initializer))
    #     H1_initializer_list.append(H1_initializer)
    #     record_list.append(tuple([loss, i]))
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # safe_print(f"To compute {number_of_random} number of random trials, the total elapsed time is : {elapsed_time} seconds")
    
    # record_list_sorted = sorted(record_list, key=lambda x: x[0])
    # H1_initializer = H1_initializer_list[record_list_sorted[0][1]]
    # print(f"Final loss = {record_list_sorted[0][0]}")
    
    # L_est = H1_initializer
    # L_est = (L_est.T / np.linalg.norm(L_est, axis = 1)).T
    # L_est = norm_multiple_stream_result(np.mat(L_est))
    # L_est = mtx2outputdata_result(L_est)
    # Score = blk.calc_score(L_est)
    

    # #*************** end of trying method similar to deep learning ********************
    
    # print("=============================")
    
    
    
    

#******************* Case 3 and case 4 work end at here ****************************


# N_target can be [1,10]
N_target = N_tar # Notices that the N_target is different for each case. Contestants program need to estimate the number of target signals first.

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
L_est = mtx2outputdata_result(L_est)
Score = blk.calc_score(L_est)
print(f"Score = {Score}")
print("Number of signal = ",N_tar)
print(h_idx)
print("number of interference =",len(h_idx))