import os, sys, shutil
import optparse
from optparse import OptionParser
import numpy as np
from time import time
from utils import str2bool
from emulator_network import EmulatorNetwork
from emulator_params import get_emulator_params
from os.path import basename, dirname, splitext


def file2list(file_name):
    import re
    if not os.path.exists(file_name):
        stop('Could not find ' + file_name)
    file = open(file_name, 'r')
    List = []
    for line in file.readlines():
        line = line.strip()
        List.append(line)
    file.close()
    return List

class progressbarClass:
    def __init__(self, finalcount, progresschar=None):
        import sys
        self.finalcount = finalcount
        self.blockcount = 0
        if not progresschar:
            self.block = chr(178)
        else:
            self.block = progresschar
        self.f = sys.stdout
        if not self.finalcount: return
        self.f.write('\n------------------ % Progress -------------------1\n')
        self.f.write('    1    2    3    4    5    6    7    8    9    0\n')
        self.f.write('----0----0----0----0----0----0----0----0----0----0\n')
        return

    def progress(self, count):
        count = min(count, self.finalcount)
        if self.finalcount:
            percentcomplete = int(round(100 * count / self.finalcount))
            if percentcomplete < 1: percentcomplete = 1
        else:
            percentcomplete = 100
        blockcount = int(percentcomplete / 2)
        if blockcount > self.blockcount:
            for i in range(self.blockcount, blockcount):
                self.f.write(self.block)
                self.f.flush()

        if percentcomplete == 100: self.f.write("\n")
        self.blockcount = blockcount
        return


class Emulator(object):

    def __init__(self,
                 train_dir=None,
                 model_dir='models',
                 out_dir='processed',
                 **params):

        # set parameters
        if model_dir is None:
            self._model_dir = 'models'
        else:
            self._model_dir = model_dir
        if out_dir is None:
            self._out_dir = 'processed'
        else:
            self._out_dir = out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        # instantiate emulator
        self._model = EmulatorNetwork(model_dir=model_dir,
                                      **params)


    def process_list(self, testlist, audiogram, ch=0, spl=60):

        # create list of files from testlist
        filelist = file2list(testlist)

        # iterate audio files and extract features
        print('\nprocessing: {}...'.format(testlist))
        t0 = time()
        nfiles = len(filelist)
        pb=progressbarClass(nfiles,"*")
        for ctr, infile in enumerate(filelist):
        
            # logging
            pb.progress(ctr+1)

            # generate neurogram
            self.process(infile, audiogram, spl=spl, ch=ch)
        dur = time() - t0
        print("total processing time: {:.2f} sec".format(dur))
        print("average processing time: {:.2f} sec".format(dur / (ctr+1)))
        
        
    def process(self, infile, audiogram, ch=0, spl=60):
        self._model.generate_neurogram(infile, audiogram, ch=ch, spl=spl)


    def load(self):
        self._model.load()


if __name__ == '__main__':

    # Parse input command line options
    parser = OptionParser()
    parser.add_option("-f","--infile", help="input audio file")
    parser.add_option("-l","--list", help="list of input audio files")
    parser.add_option("-d","--outdir", help="directory for output files")
    parser.add_option("-s","--spl", help="spl level at which to process signal (dB)")
    parser.add_option("-a","--audiogram", help="csv file containing audiogram")
    parser.add_option("-c","--channel", help="channel to process")
    parser.add_option("-v","--model_version")
    (Options, args) = parser.parse_args()
    test_file = Options.infile
    test_list = Options.list
    out_dir = Options.outdir
    spl = Options.spl
    if spl is None:
        spl = 60
    else:
        spl = float(spl)
    audiogram = Options.audiogram
    ch = Options.channel
    if ch not in ['0','1']:
        ch = 0
    else:
        ch = int(ch)
    model_version = Options.model_version
    if model_version is None:
        model_version = 1
    elif model_version == '2':
        model_version = 2
    else:
        model_version = 1

    # get params
    params = get_emulator_params(model_version=model_version)

    # instantiate emulator class
    emulator = Emulator(out_dir=out_dir,
                        **params)
    
    # do enhancement
    emulator.load()
    if test_file is not None:
        emulator.process(test_file, audiogram, spl=spl, ch=ch)
    else:
        emulator.process_list(test_list, audiogram, spl=spl, ch=ch)
        
