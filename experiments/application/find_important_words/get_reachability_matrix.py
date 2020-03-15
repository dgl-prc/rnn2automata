import sys

sys.path.append("../../../")
import re
import subprocess
from utils.constant import *
import shutil
import numpy as np
from utils.time_util import folder_timestamp


def _prepare_prism_data(pm_file, num_prop):
    ''' The task of this func is to 1) make every state the initial state.
                                    2) change the label name.

    total_states:
    pm_file:
    :return:
    '''
    ROOT_FOLDER = os.getcwd()
    folder_id = folder_timestamp()
    tmp_data_path = os.path.join(ROOT_FOLDER, folder_id)
    os.makedirs(tmp_data_path)
    with open(pm_file, "r") as fr:
        raw_pm_lines = fr.readlines()
    total_lines = len(raw_pm_lines)
    label_begin = raw_pm_lines.index("endmodule\n") + 2
    label_patter = re.compile(r".*\"(\w+)\".*")
    for idx in range(label_begin, total_lines):
        line = raw_pm_lines[idx]
        new_label = "L" + label_patter.match(line).group(1)
        raw_pm_lines[idx] = re.sub(r"\"(\w+)\"", '\"' + new_label + '\"', line)

    ptn = re.compile(r".*\[1\.\.(\d+)\].*")
    total_states = int(ptn.match(raw_pm_lines[3]).group(1))
    for start_s in range(1, total_states + 1):
        for prop_id in range(1, num_prop + 1):
            file_name = "s{}_p{}.pm".format(start_s, prop_id)
            raw_pm_lines[3] = "s:[1..{}] init {};\n".format(total_states, start_s)
            with open(os.path.join(tmp_data_path, file_name), "w") as fw:
                fw.writelines(raw_pm_lines)
    print("Total states:{}".format(total_states))
    print("Model-check files saved in {}".format(tmp_data_path))
    return total_states, tmp_data_path


def _get_reachability_prob(prism_script, data_path, start_s, prop_id):
    pm_file = os.path.join(data_path, "s{}_p{}.pm".format(start_s, prop_id))
    property_file = get_path(PROPERTY_FILE)
    prism_command = " ".join([prism_script, pm_file, property_file, "-prop", str(prop_id)])
    output = subprocess.getoutput(prism_command)
    output = output.split("\n")
    rst = float(output[-2].split()[1])
    return rst


def _get_matrix(total_states, num_prop, tmp_prism_data_path):
    matrix = []
    for start_s in range(1, total_states + 1):
        row = []
        for pro_id in range(1, num_prop + 1):  # must be 1-index
            ele = _get_reachability_prob(PRISM_SCRIPT, tmp_prism_data_path, start_s, pro_id)
            row.append(ele)
        matrix.append(row)
    matrix = np.array(matrix)
    return matrix


def reachability_matrix(pm_file, num_prop):
    ''' calculate the matrix of label reachability.

    Parameters
    -------------------
    pm_file: str. The path of original pm model file
    num_prop: int. Number of property which need to satisfy.
    Return: ndarray. shape(num_states, num_prop)
    '''
    ##################################
    # prepare the pair of input files
    ##################################
    num_states, tmp_data_path = _prepare_prism_data(pm_file, num_prop)
    matrix = _get_matrix(num_states, num_prop, tmp_data_path)
    ##################################
    # remove the prepared files
    ##################################
    shutil.rmtree(tmp_data_path)
    print("Removed temp files in {}".format(tmp_data_path))
    return matrix
