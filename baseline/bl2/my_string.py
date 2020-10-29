class MyString(object):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data
        self.p = 0
        self.length = len(self.data)

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.p < self.length:
            ele = self.data[self.p]
            self.p += 1
        else:
            self.p = 0
            raise StopIteration()
        return ele

    def __str__(self):
        return " ".join(self.data)

    def __add__(self, other):
        if isinstance(other, str):
            if other == "":
                data = self.data
            else:
                data = self.data + [other]
        else:
            if other.data == [""]:
                data = self.data
            else:
                data = self.data + other.data

        return MyString(data)

    def __eq__(self, other):

        if not None == other:
            if isinstance(other,str):
                return " ".join(self.data) == other
            else:
                return " ".join(self.data) == " ".join(other.data)
        else:
            return False

    def __hash__(self):
        return hash(self.__str__())

    def __getitem__(self, index):
        return self.data[index]



# if __name__ == '__main__':
#     a = MyString(["","12"])
#     print(isinstance(a, Iterable))