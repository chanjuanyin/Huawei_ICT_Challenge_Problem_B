from blackbox import BloackBox
from util import mtx2outputdata, zero_rand
import numpy as np


class Solution:
    
    def __init__(self, in_file: str, ans_file: str):
        self.bx = BloackBox(in_file, ans_file)
    
    def run(self):
        self.N_intf = -1
        self.N_tar = -1
        self.find_row_ratio()
        self.find_interference_signal()
    
    def find_row_ratio(self):
        matrix_10= np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
        matrix_10[0,0] = 1.0 + 0.0*1j
        matrix_10_input = mtx2outputdata(matrix_10)
        self.y_10 = self.bx.blackboxSystem(matrix_10_input, matrix_10_input)
        
        self.ratio_matrix = np.mat(np.zeros((32, 300), dtype=complex))
        self.ratio_matrix[0] = (self.y_10 / self.y_10)[0]
        
        for i in range(1, 32):
            matrix_01 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            matrix_01[0,i] = 1.0 + 0.0*1j
            matrix_01_input = mtx2outputdata(matrix_01)
            y_01 = self.bx.blackboxSystem(matrix_01_input, matrix_01_input)
            self.ratio_matrix[i] = (y_01 / self.y_10)[0]
        return self.ratio_matrix

    def test_large(self, vector, order = 0.1):
        return np.sum(np.abs(vector[0])) > order * 300
        
    def find_interference_signal(self):
        self.h_idx = []
        test_order = 10
        if self.test_large(self.y_10[0], order = test_order):
            self.h_idx.append(0)
        for i in range(1, 32):
            matrix_01 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
            matrix_01[i,0] = 1.0 + 0.0*1j
            matrix_01_input = mtx2outputdata(matrix_01)
            y_01 = self.bx.blackboxSystem(matrix_01_input, matrix_01_input)
            if self.test_large(y_01[i], order = test_order):
                self.h_idx.append(i)
            else:
                if self.N_tar != -1: assert np.linalg.matrix_rank(y_01) == self.N_tar
                else: self.N_tar = np.linalg.matrix_rank(y_01)
            self.N_intf = len(self.h_idx)
            print(i,  np.linalg.matrix_rank(y_01))
        return self.h_idx
    
    def one_trial(self):
        w1 = np.mat(np.random.rand(32, 32) + 1j * np.random.rand(32, 32))
        w1 = (w1.T / np.linalg.norm(w1)).T
        w2 = (np.mat(np.ones((32, 32))) + 1j * np.mat(np.zeros((32, 32)))) * (1/32)
        w1_input = mtx2outputdata(w1)
        w2_input = mtx2outputdata(w2)
        return self.bx.blackboxSystem(w1_input, w2_input)
    
    def find_rank(self):
        y_trial = self.one_trial()
        idx = [x for x in range(32) if x not in self.h_idx]
        print(idx)
        y_dropped = y_trial[idx]
        print(y_dropped.shape)
        self.rank = np.linalg.matrix_rank(y_dropped)

        norms = np.linalg.norm(y_dropped, axis=0)
        smallest_norm_indices = np.argsort(norms)[:self.rank]
        columns_with_smallest_norm = y_dropped[:, smallest_norm_indices]

        return self.rank, columns_with_smallest_norm
    

    def rank_10(self):
        return np.linalg.matrix_rank(self.y_10)
    


if __name__ == "__main__":
    in_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/3.in"
    ans_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/3.ans"

    sol = Solution(in_file, ans_file)
    sol.run()
    print(sol.ratio_matrix.shape)
    print(sol.h_idx)
    print(sol.N_intf, "number of target:",sol.N_tar)