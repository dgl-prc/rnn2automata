

def load_data(data_path, symbols_count, start_symbol=None):
    '''
    The data file should comply the following rules:
    1. each line is a sequence
    2. within a sequence, each element should be split by a comma
    DataParameters.
    ----------------------------------
    data_path:
    symbols_count: int. the total number of symbols to select.
    start_symbol: None if not specify a start symbol
    Return:
        seq_list. list. the sequence list
        alphabet. set. alphabet of selected sequences
    '''
    seq_list = []
    alphabet = set()
    cnt = 0
    with open(data_path, 'rt') as f:
        for line in f.readlines():
            line = line.strip().strip("'").strip(",")
            seq = line.split(",")
            remain_len = symbols_count - cnt
            if remain_len >= len(seq):
                cnt += len(seq)
            else:
                seq = seq[:remain_len]
                cnt += remain_len
            seq = [start_symbol] + seq if start_symbol is not None else seq
            seq_list.append(seq)
            alphabet = alphabet.union(set(seq))
            if symbols_count != -1 and cnt >= symbols_count:
                break
    if cnt < symbols_count:
        print("no enough data, actual load {} symbols".format(cnt))
    return seq_list, alphabet
