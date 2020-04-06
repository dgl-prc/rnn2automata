import sys

sys.path.append("../../")
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils.help_func import *
from utils.constant import *
from data.text_utils import *
from data.text_utils import filter_stop_words
from collections import defaultdict


def length_count(X, use_clean):
    lens = ["len_{}".format(length) for length in range(10, 110, 10)]
    lens_cnt = defaultdict(list)
    assert use_clean
    for x in X:
        x = filter_stop_words(x)
        x_len = len(x)
        if x_len <= 10:
            lens_cnt["len_10"].append(x_len)
        if x_len <= 20:
            lens_cnt["len_20"].append(x_len)
        if x_len <= 30:
            lens_cnt["len_30"].append(x_len)
        if x_len <= 40:
            lens_cnt["len_40"].append(x_len)
        if x_len <= 50:
            lens_cnt["len_50"].append(x_len)
        if x_len <= 60:
            lens_cnt["len_60"].append(x_len)
        if x_len <= 70:
            lens_cnt["len_70"].append(x_len)
        if x_len <= 80:
            lens_cnt["len_80"].append(x_len)
        if x_len <= 90:
            lens_cnt["len_90"].append(x_len)
        if x_len <= 100:
            lens_cnt["len_100"].append(x_len)
    for size in lens:
        print("{}:{},avg_len:{}".format(size, len(lens_cnt[size]), int(np.average(lens_cnt[size]))))


def full_imdb(word_vectors):
    x, y = [], []
    source_folder = get_path(DataPath.IMDB.RAW_DATA)
    train_neg_path = os.path.join(source_folder, 'train', 'neg')
    train_pos_path = os.path.join(source_folder, 'train', 'pos')
    train_neg_files = os.listdir(train_neg_path)
    train_pos_files = os.listdir(train_pos_path)

    test_neg_path = os.path.join(source_folder, 'test', 'neg')
    test_pos_path = os.path.join(source_folder, 'test', 'pos')
    test_neg_files = os.listdir(test_neg_path)
    test_pos_files = os.listdir(test_pos_path)

    pos_files = [os.path.join(train_pos_path, file_name) for file_name in train_pos_files] + [
        os.path.join(test_pos_path, file_name) for file_name in test_pos_files]
    neg_files = [os.path.join(train_neg_path, file_name) for file_name in train_neg_files] + [
        os.path.join(test_neg_path, file_name) for file_name in test_neg_files]

    for file_name in pos_files:
        with open(file_name, 'r', encoding="utf-8") as f:
            line = f.readline()
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(1)

    for file_name in neg_files:
        with open(file_name, 'r', encoding="utf-8") as f:
            line = f.readline()
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y, random_state=2020)
    test_idx = len(x) // 10 * 8
    data = set_data(x, y, test_idx)
    wv_matrix = make_wv_matrix(data, word_vectors)
    save_path = get_path("data/training_data/imdb/full_data_50K.pkl")
    full_data = {}
    full_data["x"] = x
    full_data["y"] = y
    full_data["vocab"] = data["vocab"]
    full_data["classes"] = data["classes"]
    full_data["word_to_idx"] = data["word_to_idx"]
    full_data["idx_to_word"] = data["idx_to_word"]
    full_data["wv_matrix"] = wv_matrix
    save_pickle(save_path, full_data)

    length_count(x, use_clean=True)


def divide_imdb():
    full_data = load_pickle(get_path("data/training_data/imdb/full_data_50K.pkl"))
    data = {}
    X = []
    Y = []
    save_path = get_path(DataPath.IMDB.PROCESSED_DATA)
    save_wv_matrix_path = get_path(DataPath.IMDB.WV_MATRIX)
    for x, y in zip(full_data["x"], full_data["y"]):
        x = filter_stop_words(x)
        if len(x) <= 50:
            X.append(x)
            Y.append(y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2020)
    data["train_x"] =  X_train
    data["train_y"] = y_train
    data["test_x"], data["test_y"] = X_test, y_test
    data["vocab"] = full_data["vocab"]
    data["classes"] = full_data["classes"]
    data["word_to_idx"] = full_data["word_to_idx"]
    data["idx_to_word"] = full_data["idx_to_word"]

    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, full_data["wv_matrix"])
    print("train_size:{},test_size:{}".format(len(y_train), len(y_test)))
