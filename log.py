from sys import stdout
from sys import stdin

class Moddel_log(object):

    def __init__(self, filename='./log/default.log', stream=stdout):
        self.terminal = stream
        self.filename = filename
        # self.log = open(self.filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.filename, 'a')
        self.log.write(message)
        self.log.close()

    def close(self):
        self.log.close()

    def flush(self):
        #TODO(gaolongc):xxxxxx
        pass


