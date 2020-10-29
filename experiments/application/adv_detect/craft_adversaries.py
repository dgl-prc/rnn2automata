import sys

sys.path.append("../../../")
import gensim
from experiments.exp_utils import select_benign_data
from target_models.model_helper import load_model, get_model_file
from target_models.classifier_adapter import Classifier
from utils.help_func import *
from utils.constant import *
from experiments.rq4.adv_detect.textbugger.textbugger_attack import TextBugger
from data.text_utils import filter_stop_words

'''
We only generate adversaries on MR dataset. After generation of adversaries, we
update the vocabulary and its corresponding word vectors.
'''

def main(model_type, data_type, word2vec_model, check_p=-1, check_point_path=None, bug_mode=TextBugger.SUB_W,
         model_path=STANDARD_PATH, save_path=STANDARD_PATH, use_clean=False):
    max_size = 1000
    device = "cpu"
    input_dim = 300
    # omit_stopws = True if model_path != STANDARD_PATH else False
    model = load_model(model_type, data_type, device, model_path)
    data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    WV_MATRIX = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))

    classifier = Classifier(model, model_type, input_dim, data["word_to_idx"], WV_MATRIX, device)
    benign_data, benign_idx = select_benign_data(classifier, data)
    textbugger = TextBugger(classifier, word2vec_model, bug_mode)
    print("begin attacking....")
    adv_rst = []
    if check_p > 0:
        check_point_data = load_pickle(check_point_path)
        adv_rst = check_point_data["ADV_RST"]
        p = check_p
        benign_data, benign_idx = benign_data[p:], benign_idx[p:]
    else:
        p = 0
    check_point_folder = make_check_point_folder("attack_{}".format(data_type), dataset=data_type, modelType=model_type)
    # Note we can make sure the order of data is fixed since the order of processed raw data is fixed.
    # In our experiments, 1000 adversarial samples is enough.
    for item, idx in zip(benign_data, benign_idx):
        try:
            sentence, label = item
            if use_clean:
                sentence = filter_stop_words(sentence)
            if classifier.get_label(sentence) == label:
                newSent, newLabel = textbugger.attack(sentence)
                if newLabel != -1:
                    adv_rst.append((idx, newSent, newLabel))
                    p += 1
                    if p >= max_size:
                        break
        except UnicodeEncodeError as e:
            continue
        sys.stdout.write("\rattacking   {:.2f}%".format(100 * p / max_size))
        sys.stdout.flush()
        if p % 50 == 0:
            check_point_data = {"ADV_RST": adv_rst}
            save_pickle(os.path.join(check_point_folder, "check_point-{}.pkl".format(p)), check_point_data)
    if save_path == STANDARD_PATH:
        save_path = getattr(getattr(Application.AEs, data_type.upper()), model_type.upper())
        save_path = get_path(save_path.format(bug_mode))

    save_pickle(save_path, adv_rst)
    print("Done! saved in {}".format(save_path))


def test_classifier():
    device = "cpu"
    data_type = DateSet.MR
    model_type = ModelType.LSTM
    model = load_model(model_type, data_type, device)
    data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    WV_MATRIX = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))
    params = {}
    params["input_size"] = 300
    params["WV_MATRIX"] = WV_MATRIX
    from target_models.model_training import test
    acc_model = test(data, model, params, device=device)
    print(acc_model)
    classifier = Classifier(model, model_type, 300, data["word_to_idx"], WV_MATRIX, device)
    select_benign_data(classifier, data)


if __name__ == '__main__':
    data_type = sys.argv[1]
    model_type = sys.argv[2]
    bug_mode = (sys.argv[3]).upper()
    use_clean = int(sys.argv[4])
    model_file = get_model_file(data_type, model_type)
    model_path = TrainedModel.NO_STOPW.format(data_type, model_type, model_file)
    save_path = Application.AEs.NO_STOPW.format(data_type, model_type, bug_mode)
    word2vec_model_path = get_path(WORD2VEC_PATH)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    main(model_type=model_type, data_type=data_type, word2vec_model=word2vec_model, bug_mode=bug_mode,
         model_path=model_path, save_path=save_path, use_clean=use_clean)

    # test_classifier()
