import random
import numpy as np
import os

def zero_rand(*args):
    return np.random.rand(*args) - 0.5
    
def generate_H(N_intf, N_target):
    # Generate intf_H1
    intf_H1 = (zero_rand(N_intf, 32) + 1j * zero_rand(N_intf, 32)) * 1e-5
    intf_H2 = (zero_rand(N_intf, 32) + 1j * zero_rand(N_intf, 32)) * 1e-5
    intf_H3 = (zero_rand(N_intf, 32) + 1j * zero_rand(N_intf, 32)) * 1e-5
    j_indexes = [(i, x) for i, x in enumerate(random.sample(list(range(32)), k = N_intf))]
    j_record = []
    
    for j_index in j_indexes:
        j = (zero_rand(1) + 1j * zero_rand(1))
        j = j / np.absolute(j)
        j = j[0]
        # print(type(j))
        # print(j.real, j.imag)
        intf_H1[j_index] = j
        intf_H2[j_index] = j
        intf_H3[j_index] = j
        j_record.append((j_index[1]+1, j))

    target_H1 = zero_rand(N_target, 32) + 1j * zero_rand(N_target, 32)
    target_H2 = zero_rand(N_target, 32) + 1j * zero_rand(N_target, 32) 
    target_H3 = zero_rand(N_target, 32) + 1j * zero_rand(N_target, 32) 
    
    h1 = np.concatenate([intf_H1, target_H1], axis=0)
    h1 = (h1.T / np.linalg.norm(h1, axis = 1)).T
    
    h2 = np.concatenate([intf_H2, target_H2], axis=0)
    h2 = (h2.T / np.linalg.norm(h2, axis = 1)).T
    
    h3 = np.concatenate([intf_H3, target_H3], axis=0)
    h3 = (h3.T / np.linalg.norm(h3, axis = 1)).T
    
    j_record = sorted(j_record, key=lambda x: x[0])
    
    return h1, h2, h3, j_record

def generate_S():
    s1 = zero_rand(32, 300) + 1j * zero_rand(32, 300)
    s1 = s1 / np.absolute(s1)
    
    s2 = zero_rand(32, 300) + 1j * zero_rand(32, 300)
    s2 = s2 / np.absolute(s2)
    return s1, s2

def generate_c():
    c = zero_rand(50, 300) * 2 + 1j * zero_rand(50, 300) * 2
    return c

def generate_ans(signal_num):
    return np.full(signal_num, 100, dtype=int)
    
def generate_data(case: int):
    if case == 1:
        N_intf = np.random.randint(low = 2, high = 5) # I think case 1 should be simpler
        N_target = 4 # I tested on Kattis
    elif case == 2:
        N_intf = np.random.randint(low = 4, high = 10) # Case 2 more difficult
        N_target = 6 # I tested on Kattis
    elif case == 3:
        N_intf = 32
        N_target = 6 # I tested on Kattis
    elif case == 4:
        N_intf = 32
        N_target = 10 # I tested on Kattis
    signal_num = N_intf + N_target
    print(f"N_intf:{N_intf}, N_target:{N_target}")
    data_H1, data_H2, data_H3, j_record = generate_H(N_intf, N_target)
    
    s1, s2 = generate_S()
    c = generate_c()
    
    ans = generate_ans(signal_num)
    
    return data_H1, data_H2, data_H3, s1, s2, c, ans, j_record

def format_inputdata(input_data):
    
    data_list = [str(x) for x in input_data.view(float).reshape(-1)]
    
    return " ".join(data_list), len(data_list)

def save_data(file_name_1, file_name_2, file_name_3, case, h1, h2, h3, s1, s2, c, ans, j_record):
    h1_string, h1_string_length = format_inputdata(h1)
    h2_string, h2_string_length = format_inputdata(h2)
    h3_string, h3_string_length = format_inputdata(h3)
    s1_string, s1_string_length = format_inputdata(s1)
    s2_string, s2_string_length = format_inputdata(s2)
    c_string, c_string_length = format_inputdata(c)
    ans_string = " ".join([str(ans[i]) for i in range(ans.shape[0])])
    j_record_length = len(j_record)
    j_record_indexes = " ".join([str(tup[0]) for tup in j_record])
    j_record_np_array = np.array([tup[1] for tup in j_record])
    j_record_string, _ = format_inputdata(j_record_np_array)
    with open(file_name_1, "w") as f:
        f.write(str(case))
        f.write("\n{}\n".format(h1_string_length-1))
        f.write(h1_string)
        f.write("\n{}\n".format(h2_string_length-1))
        f.write(h2_string)
        f.write("\n{}\n".format(h3_string_length-1))
        f.write(h3_string)
        f.write("\n{}\n".format(c_string_length-1))
        f.write(c_string)
        f.write("\n{}\n".format(s1_string_length-1))
        f.write(s1_string)
        f.write("\n{}\n".format(s2_string_length-1))
        f.write(s2_string)
        f.write("\n")
        f.close()
    with open(file_name_2, "w") as f:
        f.write(str(ans_string))
        f.write("\n")
        f.close()
    with open(file_name_3, "w") as f:
        f.write(str(case))
        f.write("\n{}".format(j_record_indexes))
        f.write("\n{}\n".format(j_record_string))
        f.close()

if __name__ == "__main__":
    directory_path = os.path.join("offline_demo", "input_directory")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    for case in range(1, 5):
        data_H1, data_H2, data_H3, s1, s2, c, ans, j_record = generate_data(case)
        save_data(os.path.join(directory_path, "{}.in".format(case)), os.path.join(directory_path, "{}.ans".format(case)), os.path.join(directory_path, "{}.meta".format(case)), case, data_H1, data_H2, data_H3, s1, s2, c, ans, j_record)