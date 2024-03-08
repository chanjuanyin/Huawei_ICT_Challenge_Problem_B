import os
import sys
from sys import stdin
import numpy as np


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


def read_inputdata(input_data):
    m = input_data.split(' ')
    complex_len = int(len(m)/2)
    n = np.zeros(shape=(complex_len),dtype='complex128')
    for ii in range(len(m)):
        m[ii] = float(m[ii])
    for ii in range(complex_len):
        n[ii] = m[2*ii] + m[2*ii+1]*1j
    return n


def exit_with_wrong_answer(message):
    print(f'Wrong Answer: {message}', file=sys.stderr)
    exit()

def exit_with_judge_error(message):
    print(f'Internal error: {message}', file=sys.stderr)
    exit(1)


def accept_with_score(score):
    print(f'Accepted! Score: {score}', file=sys.stderr)
    exit()


class BloackBox:
    
    def __init__(self, in_file: str, ans_file: str):
        self.in_file = in_file
        self.ans_file = ans_file
        self.data_para = {}
        self._init_in(in_file)
        self._init_ans(ans_file)
        
    def _init_in(self, in_file):
        try:
            input_in = open(in_file, 'r').read().strip()
            split_data = input_in.split('\n') #Split data by \n
            self.case = int(split_data[0]) # Case index
            if not 1 <= self.case <= 4:
                exit_with_judge_error(f'Invalid data haha in input file: {case}')
            #Read preloaded data in 1.in
            dataDL0 = read_inputdata(split_data[2])
            dataDL1 = read_inputdata(split_data[4])
            dataUL = read_inputdata(split_data[6])
            data_calib = read_inputdata(split_data[8])
            data_tmpDL0 = read_inputdata(split_data[10])
            data_tmpDL1 = read_inputdata(split_data[12]) 
            self.signal_num = int(len(dataDL0)/32)
            # Reshape data to matrices
            self.data_para = {}
            self.data_para['DL_ch_set_DL0'] = np.mat(dataDL0.reshape(self.signal_num,32)).T
            self.data_para['DL_ch_set_DL1'] = np.mat(dataDL1.reshape(self.signal_num,32)).T
            self.data_para['DL_ch_set_UL'] = np.mat(dataUL.reshape(self.signal_num,32)).T
            self.data_para['calib_c'] = np.mat(data_calib.reshape(50,300)).T
            self.data_para['DL0_datatmp'] = np.mat(data_tmpDL0.reshape(32,300)).T
            self.data_para['DL1_datatmp'] = np.mat(data_tmpDL1.reshape(32,300)).T
        except Exception as e:
            exit_with_judge_error(f'Invalid data in input file {e}')
    
    def _init_ans(self, ans_file):
        m = [int(x) for x in open(ans_file, 'r').read().strip().split(' ')]
        if not 0 <= int(m[0]) <= 200:
            exit_with_judge_error(f'Invalid data in ans file: {m}')
        for ii in range(self.signal_num):
            m[ii] = int(m[ii])
        m = np.array(m)
        calib_c = self.data_para['calib_c'][:,:self.signal_num]
        self.data_para['calib_c'] = calib_c*np.diag(m)
    
            
    def blackboxSystem(self, input_1, input_2):

        # ul_channel_all = []
        # dl0_channel_all = []
        # ul_Nullproj = []

        # score = 0
        try:
            m = input_1.strip().split(' ')#Split by ' '
            if m[0] == 'END': # When program end, calculate score
                safe_print('Roger that') # Tell submission program, ready to receive estimated data now
                self.calc_score() # Calculate score
        except Exception as e:
            exit_with_wrong_answer(f'Protocol error, last write was {e}')
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

        rx_signal_tot_freq = self.whole_system(DL0_bf,DL1_bf) # output matrix to submission program. 300X32.
        
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
        # return output
        
        return rx_signal_tot_freq
        #safe_print(output)
        
    
    def whole_system(self, DL0_bf,DL1_bf):
        DL0_freq = np.zeros(shape=(int(300), 32), dtype='complex128')
        bf_tmp = DL0_bf


        bfsize0 = bf_tmp.shape[0]


        DL0_datatmp_local = self.data_para['DL0_datatmp'][:,0:bfsize0]
        DL0_datatmp_local = np.mat(DL0_datatmp_local)
        DL0_freq = np.sqrt(32)*1/np.sqrt(bfsize0)*DL0_datatmp_local*np.conj(bf_tmp)  
        
        DL1_freq = np.zeros(shape=(int(300), 32), dtype='complex128')
        bf_tmp = DL1_bf

        bfsize1 = bf_tmp.shape[0]

        
        DL1_datatmp_local = self.data_para['DL1_datatmp'][:,0:bfsize1]
        DL1_datatmp_local = np.mat(DL1_datatmp_local)
        DL1_freq = np.sqrt(32)*1/np.sqrt(bfsize1)*DL1_datatmp_local*np.conj(bf_tmp)  

        sig0_signal = DL0_freq*self.data_para['DL_ch_set_DL0']
        sig1_signal = DL1_freq*self.data_para['DL_ch_set_DL1']
        UL_signal_rate = np.multiply(np.multiply(np.multiply(sig0_signal, sig0_signal), np.conj(sig1_signal)),self.data_para['calib_c'])    
            
        rx_signal_freq = np.zeros(shape=(UL_signal_rate.shape[0],32,UL_signal_rate.shape[1]), dtype = "complex128")
        for ii in range(UL_signal_rate.shape[1]):
            rx_signal_freq[:,:,ii] = UL_signal_rate[:,ii]*self.data_para['DL_ch_set_UL'][:,ii].T
        rx_signal_tot_freq = UL_signal_rate * self.data_para['DL_ch_set_UL'].T
        
        return rx_signal_tot_freq


    def calc_score(self, input_data):
        signal_num_targ_dict = {
            1: 4,
            2: 6,
            3: 6,
            4: 10
        }
        signal_num_targ = signal_num_targ_dict[self.case]
        try:
            m = input_data.strip().split(' ')#list ['XXXX','XXXX',...]  len 128
        except Exception as e:
            exit_with_wrong_answer(f'Protocol error, last write in score was {e}')

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
        # if signal_num_targ == 2:
        #     after_proj_power = dl_Nullproj*self.data_para['DL_ch_set_DL0']
        # else:
        #     exit_with_judge_error(f'Invalid signal num : {signal_num}')
        after_proj_power = dl_Nullproj*self.data_para['DL_ch_set_DL0']
        result = 0

        for ii in range(signal_num_targ):
            result_tmp = np.linalg.norm(after_proj_power[:,ii])**2
            result = result + result_tmp

        result = 15*(1 - result/signal_num_targ)
        result = np.round(result,decimals=6)
        accept_with_score(result) #The submission code need this line
        return result
