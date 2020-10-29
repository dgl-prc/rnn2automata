import sys

sys.path.append("../")
from level2_abstract.aalergia import *
from level2_abstract.read_seq import *
from utils.constant import *


def do_L2_abs(data_type, model_type, k):
    # k = int(sys.argv[1])
    # k = 10
    alpha = 64
    total_symbols = 1000000
    partition_type = PartitionType.KM
    using_train_set = True
    data_source = "train" if using_train_set else "test"
    l1_data_path = AbstractData.Level1.NO_STOPW.format(data_type, model_type, k, data_source + ".txt")
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
    return al.total_states


if __name__ == '__main__':
    # data_type = sys.argv[1]
    # model_type = sys.argv[2]
    for _data_type in [DateSet.BP, DateSet.Tomita1, DateSet.Tomita2, DateSet.Tomita3,
                       DateSet.Tomita4, DateSet.Tomita5,
                       DateSet.Tomita6, DateSet.Tomita7, DateSet.MR, DateSet.IMDB]:
        for _model_type in [ModelType.LSTM, ModelType.GRU]:
            for _k in range(11, 21, 2):
                model_size = do_L2_abs(_data_type, _model_type, _k)
                if model_size > 100:
                    print(">>>>>>>>>>>OVERSIZE!<<<<<<<<<<<<<")
                    break
