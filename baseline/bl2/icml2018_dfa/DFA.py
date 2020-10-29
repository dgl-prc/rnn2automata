# coding:utf8
from baseline.bl2.my_string import MyString


class DFA:
    def __init__(self, obs_table, real_sense):
        self.alphabet = obs_table.A  # alphabet 相当于action集合
        self.Q = [s for s in obs_table.S if
                  s == obs_table.minimum_matching_row(s)]  # avoid duplicate states. All states
        if real_sense:
            self.q0 = obs_table.minimum_matching_row(MyString(["$"]))  # initial state
        else:
            self.q0 = obs_table.minimum_matching_row("")  # initial state
        self.F = [s for s in self.Q if obs_table.T[s] == 1]  # accept states
        self._make_transition_function(obs_table)

    def _make_transition_function(self, obs_table):
        self.delta = {}  # self.delta[s][a] denotes the **next state** leading from start state *s* with the action *a*
        for s in self.Q:
            self.delta[s] = {}
            for a in self.alphabet:
                self.delta[s][a] = obs_table.minimum_matching_row(s + a)

    def classify_word(self, word, real_sense=False):
        # assumes word is string with only letters in alphabet
        q = self.q0
        for a in word:
            if real_sense and a == "$":  # due to "$" is not in the alphabet
                continue
            q = self.delta[q][a]
        return q in self.F

    def minimal_diverging_suffix(self, state1, state2, real_sense):
        # 寻找一个后缀s，能够使得state1+s 和 state2+s 的dfa.classify_word结果不一致
        # gets series of letters showing the two states are different,
        # i.e., from which one state reaches accepting state and the other reaches rejecting state
        # assumes of course that the states are in the automaton and actually not equivalent
        res = None
        # just use BFS til you reach an accepting state
        # after experiments: attempting to use symmetric difference on copies with s1,s2 as the starting state, or even
        # just make and minimise copies of this automaton starting from s1 and s2 before starting the BFS,
        # is slower than this basic BFS, so don't
        seen_states = set()
        if real_sense:
            new_states = {(MyString(["$"]), (state1, state2))}
        else:
            new_states = {("", (state1, state2))}
        while len(new_states) > 0:
            prefix, state_pair = new_states.pop()
            s1, s2 = state_pair
            if len([q for q in [s1, s2] if
                    q in self.F]) == 1:  # intersection of self.F and [s1,s2] is exactly one state,
                # meaning s1 and s2 are classified differently
                res = prefix
                break
            seen_states.add(state_pair)
            for a in self.alphabet:
                next_state_pair = (self.delta[s1][a], self.delta[s2][a])
                next_tuple = (prefix + a, next_state_pair)
                if not next_tuple in new_states and not next_state_pair in seen_states:
                    new_states.add(next_tuple)
        if real_sense:
            if len(res) > 1 and res[0] == "$":
                res = MyString(res[1:])  # 后缀不能有$
        return res
