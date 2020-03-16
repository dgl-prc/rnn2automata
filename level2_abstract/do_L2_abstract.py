import sys
sys.path.append("../")
from level2_abstract.aalergia import *
from level2_abstract.read_seq import *
from utils.constant import *

def do_L2_abs():
    # k = int(sys.argv[1])

    using_train_set = False
    partition_type = PartitionType.KM
    model_type = ModelType.LSTM
    data_type = DateSet.MR
    total_symbols = 500000

    data_source = "train" if using_train_set else "test"
    temp1 = getattr(AbstractData.Level2, partition_type.upper())
    output_folder = get_path(getattr(getattr(temp1, model_type.upper()), data_type.upper()))
    output_path = os.path.join(output_folder, data_source)

    for k in range(2, 4, 2):
        print("***********k={}*********".format(k))
        data_path = get_L1_data_path(partition_type, model_type, data_type, data_source, k)
        sequence, alphabet = load_trace_data(data_path, total_symbols)
        print(current_timestamp(), "init")
        al = AALERGIA(64, sequence, alphabet, start_symbol=START_SYMBOL, output_path=output_path, show_merge_info=False)
        print(current_timestamp(), "learing....")
        dffa = al.learn()
        print(current_timestamp(), "done")
        al.output_prism(dffa, "{}_{}_k{}_".format(model_type, data_type, k))


if __name__ == '__main__':
    do_L2_abs()
