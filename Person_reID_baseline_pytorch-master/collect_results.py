import subprocess
import os
import re
import argparse
print("Collecting results")
parser = argparse.ArgumentParser(description='results extraction')
parser.add_argument('--model_name',default='image_net_erasing', type=str, help='output model name')
parser.add_argument('--epoch_sets', default=5, type=int, help='batchsize')
opt = parser.parse_args()
model_name = opt.model_name
model_dir = "model"+"/"+model_name
print("model: ", model_name)
result_string = "";
def get_epoch_list(model_name):
    files_list = os.listdir(model_dir)
    epochs_list = []
    for name in files_list:
        if "net_" in name and ".pth" in name:
            num = ((name.split("net_"))[1].split(".pth")[0])
            epochs_list.append(num)
    return epochs_list

def get_epoch_results(model_name):
    global result_string
    f = open(model_dir + "/epoch_results.txt", "w")
    epochs_list = get_epoch_list(model_name)
    if('last' in epochs_list): 
        epochs_list.remove('last')
        epochs_list.sort(key=lambda x: int(x), reverse=True)
        epochs_list = ['last'] + epochs_list
    else:
        epochs_list.sort(key=lambda x: int(x), reverse=True)
    for epoch_no in epochs_list[0:opt.epoch_sets]:
        epoch_str = str(epoch_no)
        command_list = ["python", "extract.py", "--which_epoch", epoch_str, "--model_name", model_name]
        subprocess.call(command_list)
        evaluate_command_list = ["python", "evaluate_int.py", "--model_name", model_name]
        result = subprocess.check_output(evaluate_command_list)
        result = " model " + model_name + "  epoch_no: " + epoch_str + " ----> " + str(result.decode("utf-8"))
        print(result)
        result_string = result_string + result;
        f.write(result)
    print(result_string)
        
get_epoch_results(model_name)