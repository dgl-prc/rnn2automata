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


def test(data, model, params, mode="test", device="cuda:0"):
    model.eval()
    if mode == "train":
        X, Y = data["train_x"], data["train_y"]
    elif mode == "test":
        X, Y = data["test_x"], data["test_y"]
    acc = 0
    for sent, c in zip(X, Y):
        input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device)
        output, _ = model(input_tensor)
        # avg_h = torch.mean(output, dim=1, keepdim=False)
        lasthn = output[0][-1].unsqueeze(0)
        pred = model.h2o(lasthn)
        label = data["classes"].index(c)
        pred = np.argmax(pred.cpu().data.numpy(), axis=1)[0]
        acc += 1 if pred == label else 0
    return acc / len(X)


def train(data, params):
    model = init_model(params)
    device = "cuda:{}".format(params["GPU"]) if params["GPU"] != 1 else "cpu"
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()
    pre_test_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        i = 0
        model.train()
        for sent, c in zip(data["train_x"], data["train_y"]):
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
        test_acc = test(data, model, params, mode="test", device=device)
        print("epoch:", e + 1, "/ test_acc:", test_acc)
        if params["EARLY_STOPPING"] and test_acc <= pre_test_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_test_acc = test_acc
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)
    max_train_acc = test(data, best_model, params, mode="train", device=device)
    print("train_acc:{0:.4f}, test acc:{1:.4f}".format(max_train_acc, max_test_acc))
    return best_model, max_train_acc, max_test_acc


def main():
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    gpu = int(sys.argv[3])

    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))()
    data = load_pickle(get_path(getattr(DataPath, dataset.upper()).PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))
    train_args.add_data_info(data, params)
    params["WV_MATRIX"] = wv_matrix
    params["GPU"] = gpu
    params["rnn_type"] = model_type

    model, train_acc, test_acc = train(data, params)

    # save model
    save_folder = getattr(getattr(TrainedModel, model_type.upper()), dataset.upper())
    save_path = os.path.join(PROJECT_ROOT, save_folder, folder_timestamp())
    save_model(model, train_acc, test_acc, abspath(save_path))
    save_readme(save_path, ["{}:{}\n".format(key, params[key]) for key in params.keys()])
    print("model saved to {}".format(save_path))


if __name__ == '__main__':
    main()
