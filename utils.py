#-------------------------------------------------------------------------------
#  Copyright (c) 2022 MIT Lincoln Laboratory.  This material may be 
#    reproduced by or for the U.S. Government pursuant to the copyright 
#    license under the clause at DFARS 252.227-7014 (Nov. 1995).  
#-------------------------------------------------------------------------------

import numpy as np
from scipy.io.wavfile import read, write


def str2bool(x):
    if x in ['TRUE', 'True', 'true', '1']:
        return True
    else:
        return False

def read_wavfile(wavfile):
    fs, sig = read(wavfile)
    sig = sig.astype('float32')
    return fs, sig
    
    
def write_wavfile(outfile, x, fs):
    write(outfile, fs, x)

    
def marks2labels(markfile, time_hop):

    # parse marks file
    marks = []
    f = open(markfile, 'r')
    for line in f:
        lab, start, dur = line.strip().split(' ')
        start, dur = float(start), float(dur)
        start_idx = int(start / time_hop)
        end_idx = int((start + dur) / time_hop)
        marks.append([lab, start_idx, end_idx])
    f.close()
    
    # create labels
    labels = np.zeros((marks[-1][2]+1,1))
    for mark in marks:
        lab, start, end = mark
        if lab == 'speech':
            labels[start:end+1] = 1

    # return labels
    return labels

def labels2marks(labels, markfile, time_hop):

    label_map = {0:'non-speech',
                 1:'speech'}

    f = open(markfile, 'w')
    old_lab, old_start = None, 0
    for ctr, lab in enumerate(labels):
        if old_lab is not None and lab != old_lab:
            start = old_start * time_hop
            dur = (ctr - old_start) * time_hop
            label = label_map[int(old_lab)]
            f.write("{} {} {}\n".format(label, start, dur))
            old_start = ctr
            old_lab = lab
        if old_lab is None:
            old_lab = lab
    label = label_map[lab]
    start = old_start * time_hop
    dur = (ctr - old_start) * time_hop
    f.write("{} {} {}\n".format(label, start, dur))
    f.close()


