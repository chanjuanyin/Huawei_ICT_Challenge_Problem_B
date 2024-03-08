import numpy as np
import types
import random
from blackbox import BloackBox as blk
#need to import yunpeng step 0
#step 0 function is finding ratio matrix, it is a 32*300 matrix

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

def find_infr(y : np.matrix) -> list:
    '''ratio_matrix is expected to be 32,300 shape
       y is expected to be 32,300 shape
       w1 is expected to be 32,32 shape, with first column have one 1, the other all 0
       w2 is expected to be 32,32 shape, and we will fixed w2 elements to 1/32'''

    infr_list = []
    
    for y_row in range(32):
        sum_row = 0
        for y_col in range(300):
            magnitude = np.linalg.norm((y[y_row,y_col]))
            sum_row += magnitude
        avg_row = sum_row / 300
        if 0.999 <= avg_row <= 1.001:
            infr_list.append(y_row)
    
    return infr_list
    

    
if __name__ == "__main__":
    in_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/1.in"
    ans_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/1.ans"  
    bx = blk(in_file, ans_file)
    indices_and_norms = []
    for i in range(32):
        matrix_W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        matrix_W1[i,0] = 1.0 + 0.0j
        matrix_W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
        input_weight_1 = mtx2outputdata(matrix_W1)
        input_weight_2 = mtx2outputdata(matrix_W1)
        y = bx.blackboxSystem(input_weight_1, input_weight_2)
        # if i==17: 
        #     print(y, "y_17") 
        magnitude = np.linalg.norm(y[i, :]) 
        indices_and_norms.append((i, magnitude)) 
    print(indices_and_norms)
        
    