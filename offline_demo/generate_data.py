import random
import numpy as np

def zero_rand(*args):
    return np.random.rand(*args) - 0.5
    
def generate_H(N_intf, N_target):
    # Generate intf_H1
    intf_H1 = (zero_rand(N_intf, 32) + 1j * zero_rand(N_intf, 32)) * 1e-5
    intf_H2 = (zero_rand(N_intf, 32) + 1j * zero_rand(N_intf, 32)) * 1e-5
    intf_H3 = (zero_rand(N_intf, 32) + 1j * zero_rand(N_intf, 32)) * 1e-5
    j_indexes = [(i, x) for i, x in enumerate(random.sample(list(range(32)), k = N_intf))]
   
    for j in j_indexes:
        intf_H1[j] = 1
        intf_H2[j] = 1
        intf_H3[j] = 1

    target_H1 = zero_rand(N_target, 32) + 1j * zero_rand(N_target, 32)
    target_H2 = zero_rand(N_target, 32) + 1j * zero_rand(N_target, 32) 
    target_H3 = zero_rand(N_target, 32) + 1j * zero_rand(N_target, 32) 
    
    h1 = np.concatenate([intf_H1, target_H1])
    h1 = (h1.T / np.linalg.norm(h1, axis = 1)).T
    
    h2 = np.concatenate([intf_H2, target_H2])
    h2 = (h2.T / np.linalg.norm(h2, axis = 1)).T
    
    h3 = np.concatenate([intf_H3, target_H3])
    h3 = (h1.T / np.linalg.norm(h3, axis = 1)).T
    
    return h1, h2, h3

def generate_S():
    s1 = zero_rand(32, 300) + 1j * zero_rand(32, 300)
    s1 = s1 / np.absolute(s1)
    
    s2 = zero_rand(32, 300) + 1j * zero_rand(32, 300)
    s2 = s2 / np.absolute(s2)
    return s1, s2

def generate_c():
    c = zero_rand(50, 300) * 2 + 1j * zero_rand(50, 300) * 2
    return c
    
def generate_data(case: int):
    if case == 1 or case == 2:
        N_intf = np.random.randint(low = 0, high = 10)
    elif case == 3 or case == 4:
        N_intf = 10
    N_target = np.random.randint(low = 0, high = 10)
    signal_num = N_intf + N_target
    print(f"N_intf:{N_intf}, N_target:{N_target}")
    data_H1, data_H2, data_H3 = generate_H(N_intf, N_target)
    
    s1, s2 = generate_S()
    c = generate_c()
    return data_H1, data_H2, data_H3, s1, s2, c

def format_inputdata(input_data):
    
    data_list = [str(x) for x in input_data.view(float).reshape(-1)]
    
    return " ".join(data_list)

def save_data(file_name, case, h1, h2, h3, s1, s2, c):
    h1_string = format_inputdata(h1)
    h2_string = format_inputdata(h2)
    h3_string = format_inputdata(h3)
    s1_string = format_inputdata(s1)
    s2_string = format_inputdata(s2)
    c_string = format_inputdata(c)
    with open(file_name, "w") as f:
        f.write(str(case))
        f.write("\n127\n")
        f.write(h1_string)
        f.write("\n127\n")
        f.write(h2_string)
        f.write("\n127\n")
        f.write(h3_string)
        f.write("\n29999\n")
        f.write(c_string)
        f.write("\n19199\n")
        f.write(s1_string)
        f.write("\n19199\n")
        f.write(s2_string)
        f.write("\n")
        

if __name__ == "__main__":
    case = 1
    data_H1, data_H2, data_H3, s1, s2, c = generate_data(case)
    save_data("2.in", 1, data_H1, data_H2, data_H3, s1, s2, c)