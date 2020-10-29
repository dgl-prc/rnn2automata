from baseline.bl2.icml2018_dfa.ObservationTable import ObservationTable
from baseline.bl2.icml2018_dfa import DFA
from time import clock


def run_lstar(teacher, time_limit, real_sense=False):
    table = ObservationTable(teacher.alphabet, teacher, real_sense=real_sense)
    start = clock()
    teacher.counterexample_generator.set_time_limit(time_limit, start)
    table.set_time_limit(time_limit, start)

    while True:
        # find an closed(? unclosed) table
        # DGL: get a stable OT
        print("==================================")
        while True:
            while table.find_and_handle_inconsistency():
                pass
            if table.find_and_close_row():  # returns whether table was unclosed
                continue
            else:
                break
        # DGL: after a stable OT got, then we build a DFA over the OT.
        dfa = DFA.DFA(obs_table=table, real_sense=real_sense)
        print("obs table refinement took " + str(int(1000 * (clock() - start)) / 1000.0))

        # DGL: we use the equivalence query to revise the model.
        try:
            counterexample = teacher.equivalence_query(dfa, real_sense)
        except RecursionError as e:
            break
        if counterexample == None:
            break
        start = clock()
        # DGL: produce a new state and update the S.
        table.add_counterexample(counterexample, teacher.classify_word(counterexample), real_sense)
    return dfa
