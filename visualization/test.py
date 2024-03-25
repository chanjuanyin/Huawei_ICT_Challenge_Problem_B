from blackbox import BloackBox
import numpy as np

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

if __name__ == "__main__":
    in_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/1.in"
    ans_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/1.ans"  
    bx = BloackBox(in_file, ans_file)
    matrix_W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
    matrix_W1[0,0] = 1.0 + 0.0j
    matrix_W2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
    input_weight_1 = mtx2outputdata(matrix_W1)
    input_weight_2 = mtx2outputdata(matrix_W1)
    Y = bx.blackboxSystem(input_weight_1, input_weight_2)
    