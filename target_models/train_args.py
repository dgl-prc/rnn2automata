def add_data_info(data, params):
    params["MAX_SENT_LEN"] = max([len(sent) for sent in data["train_x"] + data["test_x"]])
    params["VOCAB_SIZE"] = len(data["vocab"])
    params["CLASS_SIZE"] = len(data["classes"])

def args_lstm_mr():
    params = {
        "input_size": 300,
        "output_size": 2,
        "hidden_size": 512,
        "num_layers": 1,
        "LEARNING_RATE": 0.01,
        "min_acc": 0.9999,
        "EPOCH": 100,
        "EARLY_STOPPING": False,
    }
    return params

def args_gru_mr():
    return args_lstm_mr()

def args_lstm_imdb():
    params = {
        "input_size": 300,
        "output_size": 2,
        "hidden_size": 512,
        "num_layers": 1,
        "LEARNING_RATE": 0.01,
        "min_acc": 0.9999,
        "EPOCH": 100,
        "EARLY_STOPPING": False,
    }
    return params

def args_gru_imdb():
    return args_lstm_mr()

def args_lstm_spam():
    params = {
        "input_size": 300,
        "output_size": 2,
        "hidden_size": 10,
        "num_layers": 3,
        "LEARNING_RATE": 0.01,
        "min_acc": 0.9999,
        "EPOCH": 100,
        "EARLY_STOPPING": False,
    }
    return params

def args_gru_spam():
    return args_lstm_mr()
