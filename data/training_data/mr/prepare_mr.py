import sys
sys.path.append("../../")
from sklearn.utils import shuffle
from utils.help_func import *
from utils.constant import *
from data.text_utils import *
import gensim

def divide_mr(word_vectors):
    x, y = [], []
    save_path = get_path(DataPath.MR.PROCESSED_DATA)
    save_wv_matrix_path = get_path(DataPath.MR.WV_MATRIX)

    pos_path = os.path.join(get_path(DataPath.MR.RAW_DATA),"rt-polarity.pos")
    neg_path = os.path.join(get_path(DataPath.MR.RAW_DATA),"rt-polarity.neg")

    with open(pos_path, "r",encoding="latin-1") as f:
        for line in f:
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(1)
    with open(neg_path, "r", encoding="latin-1") as f:
        for line in f:
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(0)
    x, y = shuffle(x, y)
    test_idx = len(x) // 10 * 8
    data = set_data(x, y, test_idx)
    wv_matrix = make_wv_matrix(data, word_vectors)

    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path,wv_matrix)
