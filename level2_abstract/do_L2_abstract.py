import sys
sys.path.append("../")
import os
from level2_abstract.aalergia import *
from level2_abstract.read_seq import *
from utils.constant import *

def do_L2_abs():
    k = int(sys.argv[1])
    total_symbols = 500000
    data_path = get_path(os.path.join(AbstractData.Level1.LSTM.MR,"k={}".format(k),"train.txt"))
    sequence, alphabet = load_data(data_path, total_symbols)
    print(current_timestamp(), "init")
    al = AALERGIA(64, sequence, alphabet, start_symbol=START_SYMBOL, output_path="./",show_merge_info=False)
    print(current_timestamp(), "learing....")
    dffa = al.learn()
    print(current_timestamp(), "done")
    al.output_prism(dffa, "lstm_mr_k{}_".format(k))

if __name__ == '__main__':
    do_L2_abs()
