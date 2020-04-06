import sys

sys.path.append("../")
from level2_abstract.aalergia import *
from level2_abstract.read_seq import *
from utils.constant import *


def do_L2_abs(k):
    # k = int(sys.argv[1])
    # k = 10
    data_type = sys.argv[1]
    model_type = sys.argv[2]
    alpha = 64
    total_symbols = 1000000
    partition_type = PartitionType.KM
    using_train_set = True
    data_source = "train" if using_train_set else "test"
    l1_data_path = AbstractData.Level1.NO_STOPW.format(data_type, model_type, k, data_source+".txt")
    output_path = AbstractData.Level2.NO_STOPW.format(data_type, model_type, k, alpha)

    if output_path == STANDARD_PATH:
        temp1 = getattr(AbstractData.Level2, partition_type.upper())
        output_folder = get_path(getattr(getattr(temp1, model_type.upper()), data_type.upper()))
        output_path = os.path.join(output_folder, data_source)

    if l1_data_path == STANDARD_PATH:
        l1_data_path = get_L1_data_path(partition_type, model_type, data_type, data_source, k)

    print("***********k={}***alpha={}***partitioner={}***".format(k, alpha, partition_type))
    sequence, alphabet = load_trace_data(l1_data_path, total_symbols)
    print("{}, init".format(current_timestamp()))
    al = AALERGIA(alpha, sequence, alphabet, start_symbol=START_SYMBOL, output_path=output_path,
                  show_merge_info=False)
    print("{}, learing....".format(current_timestamp()))
    dffa = al.learn()
    print("{}, done.".format(current_timestamp()))
    al.output_prism(dffa, data_source)


if __name__ == '__main__':
    for _k in range(2, 22, 2):
        do_L2_abs(_k)
