class BadInput(Exception):
    def __init__(self,msg):
        super(BadInput, self).__init__(msg)