from flask import Flask, render_template, request
import numpy as np
from blackbox import BloackBox
# from calculate_Y import calculate_Y  # Import the function from calculate_Y.py

def mtx2outputdata(input_data):
    input_data = np.mat(input_data)
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

def calculate_Y(matrix_W1, matrix_W2):
    in_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/1.in"
    ans_file = "/Users/zhangsiwei/Desktop/NTU/coding projects/huawei/Huawei_ICT_Challenge_Problem_B/offline_demo/input_directory/1.ans"
    bx = BloackBox(in_file, ans_file)
    input_weight_1 = mtx2outputdata(matrix_W1)
    input_weight_2 = mtx2outputdata(matrix_W2)
    Y = bx.blackboxSystem(input_weight_1, input_weight_2)
    # print(Y.shape)
    y_matrices["Y"] = np.array(Y).reshape(32, 300)
    y_matrices["S1"] = np.array(bx.data_para["DL0_datatmp"])
    y_matrices["S2"] = np.array(bx.data_para["DL1_datatmp"])
    return y_matrices["Y"]

app = Flask(__name__)

# Initial values for matrices W1, W2, and Y
matrix_W1 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
matrix_W1[0,13] = 1.0 + 0.0j
matrix_W2 = np.mat(np.zeros((32, 32))) + 1j * np.mat(np.zeros((32, 32)))
matrix_W2[0,13] = 1.0 + 0.0j
matrix_Y = np.zeros((32, 300), dtype=complex)
selected_Y = 'Y'  # Default selected Y matrix
y_matrices = {
    'Y': np.zeros((32, 300), dtype=complex),
    'S1': np.zeros((42, 200), dtype=complex),
    'S2': np.zeros((42, 400), dtype=complex)
}

def assign_color(value):
    try:
        log_value = np.log10(abs(value))  # Taking the logarithm of the absolute value
        if np.isnan(log_value) or np.isinf(log_value):
            return "rgb(255,255,255)"  # White color for NaN or infinite values
        # Map log values to RGB colors
        red = int(150 + log_value * 50)
        green = 150
        blue = 150
        # print(f"rgb({red},{green},{blue})")
        return f"rgb({red},{green},{blue})"
    except (ValueError, TypeError):
        return "rgb(255,255,255)"  # White color for non-numeric or complex values

@app.route('/')
def index():
    matrix_Y = y_matrices[selected_Y]
    colors_Y = [[assign_color(value) for value in row] for row in matrix_Y]
    colors_Y = np.array(colors_Y).reshape(matrix_Y.shape)
    return render_template('index.html',  matrix_W1=matrix_W1, matrix_W2=matrix_W2, y_matrices=y_matrices, selected_Y=selected_Y, colors_Y=colors_Y)

@app.route('/update_matrices', methods=['POST'])
def update_matrices():
    global matrix_W1, matrix_W2, matrix_Y, selected_Y
    matrix_W1 = np.array([[complex(request.form[f'matrix_W1_{i}_{j}']) for j in range(32)] for i in range(32)])
    matrix_W2 = np.array([[complex(request.form[f'matrix_W2_{i}_{j}']) for j in range(32)] for i in range(32)])
    matrix_Y = calculate_Y(matrix_W1, matrix_W2)
    selected_Y = request.form['selected_Y']
    return index()

if __name__ == '__main__':
    app.run(debug=True)
