import sys

sys.path.append("../../")
import gensim
from target_models.model_helper import load_model
from target_models.classifier_adapter import Classifier
from utils.help_func import *
from utils.time_util import *
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


def select_benign_data(classifier, data):
    acc = 0
    benigns = []
    idx = []
    i = 0
    for sent, label in zip(data["test_x"], data["test_y"]):
        pred = classifier.get_label(sent)
        if pred == label:
            acc += 1
            benigns.append((sent, label))
            idx.append(i)
        i += 1
    print("Acc:{:.4f}".format(acc / len(data["test_x"])))
    return benigns, idx


def main(model_type, data_type, word2vec_model, check_p=-1, check_point_path=None):
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
        benign_data, benign_idx = benign_data[:p], benign_idx[:p]
    else:
        p = 0
    total = len(benign_idx)
    check_point_folder = make_check_point_folder("attack_{}".format(data_type), dataset=data_type, modelType=model_type)
    # Note we can make sure the order of data is fixed since the order of processed raw data is fixed.
    # In our experiments, 1000 adversarial samples is enough.
    for item, idx in zip(benign_data[:1200], benign_idx[:1200]):
        progress = "{} progress:{}/{}".format(current_timestamp(), p, total)
        p += 1
        try:
            sentence, label = item
            if classifier.get_label(sentence) == label:
                newSent, newLabel = textbugger.attack(sentence)
                if newLabel != -1:
                    print(progress, " ".join(newSent), newLabel)
                    adv_rst.append((idx, newSent, newLabel))
                else:
                    print(progress, ">>>>>>>>Failed!")
        except UnicodeEncodeError as e:
            print(progress, ">>>>>>>>Failed!")
            continue

        if p % 50 == 0:
            check_point_data = {"ADV_RST": adv_rst,
                                "VOCAB": classifier.VOB,
                                "IDX2WORD": classifier.IDX2WORD,
                                "WORD2IDX": classifier.WORD2IDX,
                                "WV_MATRIX": classifier.WV_MATRIX
                                }
            save_pickle(os.path.join(check_point_folder, "check_point-{}.pkl".format(p)), check_point_data)

    ADV_DATA_PATH = getattr(getattr(AdvDataPath, data_type.upper()), model_type.upper())
    save_pickle(get_path(ADV_DATA_PATH.ADV_TEXT), adv_rst)
    ##################################################
    # save changed vocab,word2idx,idx2word,wv_matrix
    ##################################################
    save_pickle(get_path(ADV_DATA_PATH.VOCAB), classifier.VOB)
    save_pickle(get_path(ADV_DATA_PATH.IDX2WORD), classifier.IDX2WORD)
    save_pickle(get_path(ADV_DATA_PATH.WORD2IDX), classifier.WORD2IDX)
    save_pickle(get_path(ADV_DATA_PATH.WV_MATRIX), classifier.WV_MATRIX)

    print("Done!")


if __name__ == '__main__':
    data_type = sys.argv[1]
    model_type = sys.argv[2]
    main(model_type=model_type, data_type=data_type, check_p=-1)

    #
