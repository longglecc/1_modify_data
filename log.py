from sys import stdout
from sys import stdin
import os

class Moddel_log(object):

    def __init__(self, filename='./log/default.log', stream=stdout):
        self.terminal = stream
        self.filename = filename
        # self.log = open(self.filename, 'a')
        # self.mkdir()

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

    def mkdir(self):

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)
            print("---  new log file has created...  ---")
            print("---  OK  ---")

        else:
            print("---  log file has been.  ---")