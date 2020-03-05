import sys
sys.path.append("../../")
from sklearn.utils import shuffle
from utils.help_func import *
from utils.constant import *
from data.text_utils import  *

def divide_imdb(word_vectors):
    x, y = [], []
    source_folder = get_path(DataPath.IMDB.RAW_DATA)
    save_path = get_path(DataPath.IMDB.PROCESSED_DATA)
    save_wv_matrix_path = get_path(DataPath.IMDB.WV_MATRIX)

    train_neg_path = os.path.join(source_folder, 'train', 'neg')
    train_pos_path = os.path.join(source_folder, 'train', 'pos')
    neg_files = os.listdir(train_neg_path)
    pos_files = os.listdir(train_pos_path)

    pos_files, neg_files = shuffle(pos_files, neg_files)

    for file_name in pos_files:
        with open(os.path.join(train_pos_path, file_name), 'r', encoding="utf-8") as f:
            line = f.readline()
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(1)

    for file_name in neg_files:
        with open(os.path.join(train_neg_path, file_name), 'r', encoding="utf-8") as f:
            line = f.readline()
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    test_idx = len(x) // 10 * 8

    data = set_data(x, y, test_idx)
    wv_matrix = make_wv_matrix(data, word_vectors)

    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, wv_matrix)


