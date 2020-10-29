# coding:utf8
from time import clock
from baseline.bl2.my_string import MyString

class TableTimedOut(Exception):
    pass


class ObservationTable:
    def __init__(self, alphabet, interface, max_table_size=None, real_sense=False):
        if real_sense:
            self.S = {MyString(["$"])}  # starts. invariant: prefix closed
            self.E = {MyString([""])}  # ends. invariant: suffix closed
        else:
            self.S = {""}  # starts. invariant: prefix closed
            self.E = {""}  # ends. invariant: suffix closed
        self.T = interface.recorded_words  # {} #finite function T: (S cup (S dot A)) dot E -> {True,False}, might also have more info if
        # interface remembers more, but this is not harmful so long as it contains what it needs
        self.A = alphabet  # alphabet
        self.interface = interface  # in this implementation, it is the teacher.
        self._fill_T()
        self._initiate_row_equivalence_cache()  # self.equal_cache 保存的是不相等的行 (row1,row2)
        self.max_table_size = max_table_size
        self.time_limit = None
        self.real_sense = real_sense

    def set_time_limit(self, time_limit, start):
        self.time_limit = time_limit
        self.start = start
        # self._Trange 给observation table 添加新的元素；self.interface.update_words 记录新元素的成员关系

    def _fill_T(self, new_e_list=None,
                new_s=None):  # modifies, and involved in every kind of modification. modification: store more words
        self.interface.update_words(self._Trange(new_e_list, new_s))

    def _Trange(self, new_e_list, new_s):  # T: (S cup (S dot A)) dot E -> {True,False} #doesn't modify
        E = self.E if new_e_list == None else new_e_list
        starts = self.S | self._SdotA() if new_s == None else [new_s + a for a in (list(self.A) + [""])]
        return set([s + e for s in starts for e in E])

    def _SdotA(self):  # doesn't modify
        return set([s + a for s in self.S for a in self.A])

    def _initiate_row_equivalence_cache(self):
        self.equal_cache = set()  # subject to change
        for s1 in self.S:  # 判断S中是否存在相等的两行，目的是精简？
            for s2 in self.S:
                for a in list(self.A) + [""]:
                    if self._rows_are_same(s1 + a, s2):  # 如果这两行相等，则将其加入到缓存里面
                        self.equal_cache.add((s1 + a, s2))

    def _update_row_equivalence_cache(self, new_e=None,
                                      new_s=None):  # just fixes cache. in case of new_e - only makes it smaller
        if not None == new_e:
            remove = [(s1, s2) for s1, s2 in self.equal_cache if not self.T[s1 + new_e] == self.T[s2 + new_e]]
            self.equal_cache = self.equal_cache.difference(remove)
        else:  # new_s != None, or a bug!
            for s in self.S:
                for a in (list(self.A) + [""]):
                    if self._rows_are_same(s + a, new_s):
                        self.equal_cache.add((s + a, new_s))
                    if self._rows_are_same(new_s + a, s):
                        self.equal_cache.add((new_s + a, s))

    def _rows_are_same(self, s, t):  # doesn't modify
        # row(s) = f:E->{0,1} where f(e)=T(se)
        # 判断S中两个元素是否不想等
        # 只要找到一个后缀能够让其相等，则返回false。
        #################################################################################
        # rewrite the following statement for the sake of debug
        # return next((e for e in self.E if not self.T[s+e]==self.T[t+e]),None) == None
        ##################################################################################
        not_same = []
        for e in self.E:
            key1 = s + e
            key2 = t + e
            if not self.T[key1] == self.T[key2]:
                not_same.append(e)
        if len(not_same) == 0:
            return True
        else:
            return False

    def all_live_rows(self):
        return [s for s in self.S if s == self.minimum_matching_row(s)]

    def minimum_matching_row(self, t):  # doesn't modify
        # to be used by automaton constructor once the table is closed
        # not actually minimum length but so long as we're all sorting them by something then whatever
        return next(s for s in self.S if (t, s) in self.equal_cache)

    def _assert_not_timed_out(self):
        if not None == self.time_limit:
            if clock() - self.start > self.time_limit:
                print("obs table timed out")
                raise TableTimedOut()  # whatever, can't be bothered rn

    def find_and_handle_inconsistency(self):  # modifies - and whenever it does, calls _fill_T
        # returns whether table was inconsistent
        maybe_inconsistent = [(s1, s2, a) for s1, s2 in self.equal_cache if s1 in self.S for a in self.A
                              if not (s1 + a, s2 + a) in self.equal_cache]  # NOTE, 一开始S只有空字符串。
        if self.real_sense:
            troublemakers = [MyString([a]) + e for s1, s2, a in maybe_inconsistent for e in
                             [next((e for e in self.E if not self.T[s1 + a + e] == self.T[s2 + a + e]), None)] if
                             not e == None]
        else:
            troublemakers = [a + e for s1, s2, a in maybe_inconsistent for e in
                             [next((e for e in self.E if not self.T[s1 + a + e] == self.T[s2 + a + e]), None)] if
                             not e == None]
        if len(troublemakers) == 0:
            return False
        self.E.add(troublemakers[0])
        self._fill_T(
            new_e_list=troublemakers)  # optimistic batching for queries - (hopefully) most of these will become relevant later
        self._update_row_equivalence_cache(troublemakers[0])
        self._assert_not_timed_out()
        return True

    def find_and_close_row(self):  # modifies - and whenever it does, calls _fill_T
        # returns whether table was unclosed
        s1a = next(
            (s1 + a for s1 in self.S for a in self.A if not [s for s in self.S if (s1 + a, s) in self.equal_cache]),
            None)
        if s1a == None:
            return False
        self.S.add(s1a)
        self._fill_T(new_s=s1a)
        self._update_row_equivalence_cache(new_s=s1a)
        self._assert_not_timed_out()
        return True

    def add_counterexample(self, ce, label, real_sense=False):  # modifies - and definitely calls _fill_T
        if ce in self.S:
            print("bad counterexample - already saved and classified in table!")
            return

        # all the prefix of ce
        if real_sense:
            new_states = [MyString(ce[0:i + 1]) for i in range(len(ce)) if not MyString(ce[0:i + 1]) in self.S]
        else:
            new_states = [ce[0:i + 1] for i in range(len(ce)) if not ce[0:i + 1] in self.S]

        self.T[ce] = label
        self.S.update(new_states)
        self._fill_T()  # has to be after adding the new states
        for s in new_states:  # has to be after filling T
            self._update_row_equivalence_cache(new_s=s)
        self._assert_not_timed_out()
