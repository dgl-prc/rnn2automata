import sys
sys.path.append("../../../")
import gensim
from experiments.exp_utils import select_benign_data
from target_models.model_helper import load_model
from target_models.classifier_adapter import Classifier
from utils.help_func import *
from utils.constant import *
from experiments.application.adv_detect.textbugger.textbugger_attack import TextBugger

'''
We only generate adversaries on MR dataset. After generation of adversaries, we
update the vocabulary and its corresponding word vectors.
'''


# class MRClassifier(ClassifierInterface):
#
#     def __init__(self, model, device, VOB, IDX2WORD, WORD2IDX, INPUT_SIZE, WV_MATRIX):
#         self.model = model.to(device)
#         self.WORD2IDX = WORD2IDX
#         self.INPUT_SIZE = INPUT_SIZE
#         self.WV_MATRIX = WV_MATRIX
#         self.VOB = VOB
#         self.IDX2WORD = IDX2WORD
#         self.device = device
#
#     def get_label(self, sent):
#         probs = self.get_probs(sent)
#         pred = np.argmax(probs)
#         return pred
#
#     def get_probs(self, sent):
#         input_tensor = sent2tensor(sent, self.device, self.WORD2IDX, self.INPUT_SIZE, self.WV_MATRIX)
#         output, _ = self.model(input_tensor)
#         lasthn = output[0][-1].unsqueeze(0)
#         logits = self.model.h2o(lasthn)
#         probs = self.model.softmax(logits)
#         return probs.squeeze().detach().numpy()
#
#     def update_vob(self, word, word_vec):
#         size_vob = len(self.VOB)
#         self.VOB.append(word)
#         self.WORD2IDX[word] = size_vob
#         self.IDX2WORD[size_vob] = word
#         self.WV_MATRIX = np.concatenate((self.WV_MATRIX, np.array([word_vec])), axis=0)


def main(model_type, data_type, word2vec_model, check_p=-1, check_point_path=None):
    max_size = 1000
    device = "cpu"
    input_dim = 300
    model = load_model(model_type, data_type, device)
    data = load_pickle(get_path(getattr(DataPath, data_type.upper()).PROCESSED_DATA))
    WV_MATRIX = load_pickle(get_path(getattr(DataPath, data_type.upper()).WV_MATRIX))

    classifier = Classifier(model, model_type, input_dim, data["word_to_idx"], WV_MATRIX, device)
    benign_data, benign_idx = select_benign_data(classifier, data)
    textbugger = TextBugger(classifier, word2vec_model)
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

    ADV_DATA_PATH = getattr(getattr(Application.AEs, data_type.upper()), model_type.upper())
    save_pickle(get_path(ADV_DATA_PATH), adv_rst)
    print("Done!")


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
    word2vec_model_path = get_path(WORD2VEC_PATH)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    main(model_type=model_type, data_type=data_type, word2vec_model=word2vec_model, check_p=950,
         check_point_path="./tmp/attack_mr/mr/lstm/check_point-950.pkl")

    # test_classifier()
