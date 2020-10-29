import gensim
from utils.constant import *
from data.training_data.imdb.prepare_imdb import divide_imdb, full_imdb
from data.training_data.mr.prepare_mr import divide_mr
from data.training_data.bp.prepare_bp import divide_bp
from data.training_data.tomita.prepare_tomita import divide_tomita

if __name__ == '__main__':
    # word2vec_model_path = get_path(WORD2VEC_PATH)
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    #      word2vec_model_path, binary=True)
    # divide_imdb(word2vec_model)
    # divide_mr(word2vec_model)
    # full_imdb(word2vec_model)
    # divide_bp()
    divide_tomita("tomita2")
    # for tomita_x in range(1, 8):
    #     divide_tomita("tomita"+tomita_x)
