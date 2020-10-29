import sys

sys.path.append("../")
import copy
from os.path import abspath
import torch.optim as optim
from sklearn.utils import shuffle
from target_models.gated_rnn import *
from target_models import train_args
from target_models.model_helper import sent2tensor, init_model
from utils.constant import *
from utils.time_util import *
from utils.help_func import load_pickle, save_model, save_readme
from data.text_utils import filter_stop_words, is_use_clean


def test(data, model, params, mode="test", device="cuda:0"):
    model.eval()
    if mode == "train":
        X, Y = data["train_x"], data["train_y"]
    elif mode == "test":
        X, Y = data["test_x"], data["test_y"]
    acc = 0
    for sent, c in zip(X, Y):
        if params["use_clean"]:
            sent = filter_stop_words(sent)
        input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device)
        output, _ = model(input_tensor)
        # avg_h = torch.mean(output, dim=1, keepdim=False)
        lasthn = output[0][-1].unsqueeze(0)
        pred = model.h2o(lasthn)
        label = data["classes"].index(c)
        pred = np.argmax(pred.cpu().data.numpy(), axis=1)[0]
        acc += 1 if pred == label else 0
    return acc / len(X)


def train(model, data, params):
    device = params["device"]
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()
    pre_metric_acc = 0
    max_metric_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        i = 0
        model.train()
        for sent, c in zip(data["train_x"], data["train_y"]):
            if params["use_clean"]:
                sent = filter_stop_words(sent)
            label = [data["classes"].index(c)]
            label = torch.LongTensor(label).to(device)
            input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device)
            optimizer.zero_grad()
            output, inner_states = model(input_tensor)
            lasthn = output[0][-1].unsqueeze(0)
            pred = model.h2o(lasthn)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print("Train Epoch: {} [{}/{}]\tLoss: {:.6f}".format(e + 1, i + 1, len(data["train_x"]), loss))
            i += 1
        train_acc = test(data, model, params, mode="train", device=device)
        test_acc = test(data, model, params, mode="test", device=device)
        print("{}\tepoch:{}\ttrain_acc:{:.4f}\ttest_acc:{:.4f}".format(current_timestamp(), e + 1, train_acc,
                                                                       test_acc))
        metric_acc = (test_acc + train_acc) / 2
        if params["EARLY_STOPPING"] and metric_acc <= pre_metric_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_metric_acc = metric_acc
        if metric_acc >= max_metric_acc:
            max_metric_acc = metric_acc
            best_model = copy.deepcopy(model)
            best_model.i2h.flatten_parameters()
    best_train_acc = test(data, best_model, params, mode="train", device=device)
    best_test_acc = test(data, best_model, params, mode="test", device=device)
    last_train_acc = test(data, model, params, mode="train", device=device)
    print("train_acc:{:.4f}, test acc:{:.4f}, last_train_acc:{}".format(best_train_acc, best_test_acc, last_train_acc))
    return best_model, best_train_acc, best_test_acc


def main():
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    gpu = int(sys.argv[3])
    use_clean = is_use_clean(dataset)
    if dataset.startswith("tomita"):
        gram_id = int(dataset[-1])
        params = getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
        data = load_pickle(get_path(getattr(DataPath, "TOMITA").PROCESSED_DATA).format(gram_id, gram_id))
        wv_matrix = load_pickle(get_path(getattr(DataPath, "TOMITA").WV_MATRIX).format(gram_id, gram_id))
    else:
        params = getattr(train_args, "args_{}_{}".format(model_type, dataset))()
        data = load_pickle(get_path(getattr(DataPath, dataset.upper()).PROCESSED_DATA))
        wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))

    train_args.add_data_info(data, params)
    params["WV_MATRIX"] = wv_matrix
    params["device"] = "cuda:{}".format(gpu) if gpu >= 0 else "cpu"
    params["rnn_type"] = model_type
    params["use_clean"] = use_clean

    model = init_model(params)
    model, train_acc, test_acc = train(model, data, params)

    # save model
    save_folder = "data/no_stopws/trained_models/{}/{}/".format(dataset, model_type)
    save_path = os.path.join(PROJECT_ROOT, save_folder, folder_timestamp())
    save_model(model, train_acc, test_acc, abspath(save_path))
    save_readme(save_path, ["{}:{}\n".format(key, params[key]) for key in params.keys() if key != "WV_MATRIX"])
    print("model saved to {}".format(save_path))


if __name__ == '__main__':
    main()
