#coding:utf8
from rnn_models.rnn_arch.gated_rnn import *
import pickle
from data_factory.tomita.generator import TomitaDataProcessor
def test():
    with open("../../rnn_models/pretrained/tomita/gru-tomita3.pkl","r") as f:
        rnn = pickle.load(f)
    with open("./RStates.pkl","r") as f:
        Rstate = pickle.load(f)


    print("Test the Bad Rstate:{}".format(rnn.get_next_RState(Rstate, "0")[1]))

    print("Classify \"10\":{}".format(rnn.classify_word("10",-1)))

    RState_init,True = rnn.get_first_RState()

    RState_empty, label_empty = rnn.get_next_RState(RState_init, "")
    print("Test the empty string with RState_init:{}".format(label_empty))

    NextRstate1, label1=rnn.get_next_RState(RState_empty, "1")
    print("Test the 1 with RState_empty:{}".format(label1))

    NextRstate0, label0=rnn.get_next_RState(NextRstate1, "0")
    print("Test the 0 with NextRstate1:{}".format(label0))

    print("end!")


def test_single_step():

    dataProcessor = TomitaDataProcessor()

    rnn = GRU2(raw_input_size=3, innder_input_dim=3, num_class=2,
               hidden_size=10, num_layers=2, dataProcessor=dataProcessor)
    h0=rnn.get_first_RState()


    #########################
    # 一次性计算所有步骤
    #########################
    tensor_sequence = rnn.dataProcessor.sequence2tensor("10", 3)
    output_all, hx_all = rnn.forward(tensor_sequence)

    ##########################
    # 分步骤计算: 只有第一步，i.e., empty的输出符合，后续的步骤全部不符合
    ##########################
    rnn_h_init,label=rnn.get_first_RState()
    tensor_sequence_empty = rnn.dataProcessor.sequence2tensor("", 3,is_single_step=True)
    output_empty, hx_empty = rnn.forward(tensor_sequence_empty,rnn.list2hx(rnn_h_init))

    tensor_sequence_1 = rnn.dataProcessor.sequence2tensor("1", 3, is_single_step=True)
    output_1,hx_1= rnn.forward(tensor_sequence_1,hx_empty)

    tensor_sequence_0 = rnn.dataProcessor.sequence2tensor("0", 3, is_single_step=True)
    output_0,hx_0= rnn.forward(tensor_sequence_0,hx_1)


    print("DONE!")









if __name__ == '__main__':
    # test()
    # test_single_step()
    import os
    with open("../../rnn_models/pretrained/tomita/test-gru-tomita2.pkl","r") as f:
        rnn = pickle.load(f)

    grammar = "tomita2"
    data_folder = "../../data/tomita/training"
    data_path = os.path.join(data_folder, "{}.pkl".format(grammar))
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    data = dataset["data"]
    print(data["0"])
    print(rnn.classify_word("0",-1))

