import os, sys, shutil, glob
from utils import *
from os.path import basename
import numpy as np
import pandas as pd
import scipy as scp
import scipy.signal as scp_sig
import scipy.io as scp_io
from fitaudiogram3 import fitaudiogram3
from filters import *
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv1D, Flatten, Multiply, Input, Layer, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import Constraint
from time import time
from time import sleep
import h5py
import tensorflow.keras.backend as K
import warnings
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore")



class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):        
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


class FilterBank(Layer):
    def __init__(self,
                 ntaps,
                 nfilts,
                 name='filterbank'):
        self._nfilts = nfilts
        self._ntaps = ntaps
        super(FilterBank,self).__init__(name=name)

        
    def build(self, input_shape):
        self._filters = self.add_weight(name='filterbank', 
                                        shape=(self._ntaps,1,self._nfilts), 
                                        initializer='normal',
                                        trainable=True)
        super(FilterBank, self).build(input_shape)
        
    def call(self, x):
        y = K.conv1d(x, self._filters, padding='same')
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0], self._num_cf)
        

class Compander(Layer):
    def __init__(self, reg=1e-6):
        self._reg = reg
        super(Compander,self).__init__()
        
    def build(self, input_shape):
        self._mu = self.add_weight(name='mu', 
                                   shape=(input_shape[-1],), 
                                   initializer='ones',
                                   constraint=Between(1e-1,1e1),
                                   trainable=True)
        super(Compander, self).build(input_shape)
        
    def call(self, x):
        y = K.sign(x) * K.pow(K.abs(x) + self._reg, self._mu)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

def Conv1DBlock(x, nchans, dim, dilation=1, act=None, name=None):
    x = Conv1D(nchans, dim, dilation_rate=dilation, padding='causal')(x)
    x = Activation(act,name=name)(x)
    return x


def GatedLinearUnit(x, nchans, dim, dilation=1, desc=None, name=None):
    filter = Conv1DBlock(x, nchans, dim, dilation=dilation, name='{}_filt'.format(name))
    gate = Conv1DBlock(x, nchans, dim, dilation=dilation, act='sigmoid',name='{}_gate'.format(name))
    if desc is not None:
        mask = Conv1DBlock(desc, nchans, 1, act='sigmoid',name='{}_desc'.format(name))
        gate = Multiply()([gate, mask])
    x = Multiply()([filter, gate])
    return x


def GatedLinearUnitBlock(x, nchans, dim, dilation=1, desc=None, name=None):
    x = GatedLinearUnit(x, nchans, dim, dilation=dilation, desc=desc, name=name)
    return x


class InstanceNormalization(Layer):
    def __init__(self):
        super(InstanceNormalization,self).__init__()
        
    def build(self, input_shape):
        self._gamma = self.add_weight(name='gamma', 
                                       shape=(input_shape[2],), 
                                       initializer='ones',
                                       trainable=True)
        self._beta = self.add_weight(name='beta', 
                                       shape=(input_shape[2],), 
                                       initializer='zeros',
                                       trainable=True)
        super(InstanceNormalization, self).build(input_shape)
        
    def call(self, x):
        x = (x - K.mean(x,1,keepdims=True)) / (K.std(x,1,keepdims=True) + 1e-3)
        x = x * self._gamma[None,None,:] + self._beta[None,None,:]
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class EmulatorNetwork(object):

    def __init__(self,
                 model_dir='models',
                 **model_params):

        # set algorithm parameters
        self._model_dir = model_dir
        self._model_dir = model_dir

        # set model parameters
        self._fs = model_params['fs']
        self._frame_rate = model_params['frame_rate']
        self._fb_nfilts = model_params['fb_nfilts']
        self._channel_depth = model_params['channel_depth']
        self._out_channels = model_params['out_channels']
        self._do_instance_norm = model_params['do_instance_norm']
        self._hi_fs = 2.0 * np.array(model_params['hi_fs']) / self._fs
        self._desc_layers = model_params['desc_layers']
        self._ihc_layers = model_params['ihc_layers']
        self._ngram_layers = model_params['ngram_layers']
        self._cnn_dim = model_params['cnn_dim']
        self._cnn_dilation_max = model_params['cnn_dilation_max']
        self._do_compression = model_params['do_compression']
        self._loss_type = model_params['loss_type']
        self._lr_init = model_params['lr']
        self._lr_warmup = model_params['lr_warmup']
        self._lr_final = model_params['lr_final']
        self._num_hours = model_params['num_hours']
        self._trainfile_repeats = model_params['trainfile_repeats']
        self._batch_size = model_params['batch_size']
        self._full_dim = int(self._fs * model_params['full_dur'])
        self._loss_weights = np.concatenate([model_params['loss_ihc_const'] * np.ones(self._out_channels),
                                             model_params['loss_ngram_const'] * np.ones(self._out_channels)],0)

        # calculate parameters   
        self._full_dim = int(self._fs * model_params['full_dur'])
        self._full_dim_out = int(self._frame_rate * model_params['full_dur'])
        self._downsample_rate = int(self._fs / self._frame_rate)
        self._win_dim = int(self._fs * model_params['win_dur'])

        # create model name
        self._create_modelname()

            
    def load(self, source_dir=None):
        if source_dir is not None:
            self.model_name = '{}/{}'.format(source_dir, self.model_name)
        self._model = self._define_network(limit_atten=True)
        self._model.load_weights(self.model_name)


    def generate_neurogram(self, infile, audiogram, ch=0, spl=60, out_dir='processed'):

        # read and pad speech signal
        x, fs = self._read_speech_file(infile)
        nfr = int(np.ceil(x.shape[0]/self._fs))
        npad = nfr * self._fs - x.shape[0]
        x = np.pad(x,(0,npad))

        # apply spl and gain
        p0 = 20E-6
        pascal = (10.**(spl / 20.)) * p0
        spl_scale = pascal / (np.std(x) + 1e-7)
        x *= spl_scale

        # window input signal
        x_win = self._window_speech(x, overlap=0.5)
        x_win = np.expand_dims(np.transpose(x_win,(1,0)),-1)
        
        # get audiogram
        cohc, cihc = self._get_audiogram(audiogram)
        hi_params = np.expand_dims(np.vstack([cohc, cihc]).T,0)
        hi_params = np.repeat(hi_params, x_win.shape[0], 0)

        # generate ihc and neurogram
        y = self._model.predict([x_win, hi_params], batch_size=self._batch_size, verbose=0)
        y = np.transpose(y,(1,0,2))[:,:-2,:]
        y = self._overlap_and_add(y, overlap=0.5)
        y /= self._loss_weights
    
        # save data
        npad2 = int(npad * self._frame_rate / self._fs)
        if npad > 0:
            x = x[:-npad]
        if npad2 > 0:
            y = y[:-npad2,:]
        ihc = y[:,:self._out_channels]
        ngram = y[:,self._out_channels:]
        scp_io.savemat('{}/{}.mat'.format(out_dir, infile.split('/')[-1][:-4]),{'waveform': x,
                                                                                'ihc' : ihc,
                                                                                'ngram' : ngram})

                                                                        
    def _define_network(self, limit_atten=False):
        
        # define input waveform
        in_feats = Input(shape=(self._full_dim, 1))
        in_dbloss = Input(shape=(self._out_channels, 2))

        # apply filterbank
        y = FilterBank(nfilts=self._fb_nfilts,
                       ntaps=self._win_dim,
                       name='filterbank')(in_feats)

        # extract descriptor embedding
        hi_emb = Flatten()(in_dbloss)
        g_emb = Compander()(K.std(y, 1))
        desc = tf.expand_dims(concatenate([hi_emb, g_emb],-1),1)
        desc = Conv1DBlock(desc, self._channel_depth, 1)
        for m in range(self._desc_layers-1):
            desc = Conv1DBlock(desc, self._channel_depth, 1)

        # pre-processing
        if self._do_instance_norm:
            y = InstanceNormalization()(y)
        if self._do_compression:
            y = Compander()(y)
        y = Conv1D(self._channel_depth, self._downsample_rate, strides=self._downsample_rate,
                   activation='swish', padding='same')(y)

        # generate IHC and Neurogram
        for m in range(self._ihc_layers + self._ngram_layers):
            dilation = m
            if self._cnn_dilation_max > 0:
                dilation = dilation % self._cnn_dilation_max
            y = GatedLinearUnitBlock(y, self._channel_depth, self._cnn_dim,
                                     dilation=(dilation+1), desc=desc,
                                     name='eGLU{}'.format(m))
            if m == (self._ihc_layers - 1):
                ihc = y
            elif m == (self._ihc_layers + self._ngram_layers - 1):
                ngram = y
        ihc = Conv1DBlock(ihc, self._out_channels, 1, act='leaky_relu')
        if self._do_compression:
            ihc = Compander()(ihc)
        ngram = Conv1DBlock(ngram, self._out_channels, 1, act='relu')
        if self._do_compression:
            ngram = Compander()(ngram)

        # define model
        out = concatenate([ihc, ngram])
        model = Model([in_feats, in_dbloss], out)
        
        # compile model
        model.compile(optimizer=Adam(),
                      loss='mean_squared_error')
        return model


    @staticmethod
    def _resample_frames(x, upsample, downsample):
        x = scp_sig.resample_poly(x, upsample, downsample, axis=0)
        return x


    def _window_speech(self, x, overlap=None):
        if len(x) >= self._full_dim:
            x2 = np.hstack([x, x[-1:-self._full_dim-1:-1]])
        else:
            x2 = np.hstack([x, np.zeros(self._full_dim)])
        if overlap is not None:
            nshift = int(self._full_dim * (1. - overlap))
        else:
            nshift = self._full_dim
        nfr = int(x.shape[0] / nshift) + 1
        feats = np.zeros((self._full_dim, nfr))
        for m in range(nfr):
            feats[:,m] = x2[(m * nshift):(m * nshift + self._full_dim)]
        return feats
    
    
    def _overlap_and_add(self, x, overlap=None):
        if len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        nwin, nfr, dim = x.shape
        if overlap is None:
            nshift = nwin
            ntaper = 0
        else:
            nshift = int(nwin * (1. - overlap))
            ntaper = int(nwin * overlap)
        y = np.zeros(((nfr - 1) * nshift + nwin, dim))
        win = np.ones(nwin)
        if ntaper > 0:
            taper = np.linspace(0,1,ntaper)
            win[:ntaper] = taper
            win[-ntaper:] = taper[::-1]
        for fr in range(nfr):
            x_fr = x[:,fr,:]
            curr_win = win.copy()
            if fr == 0 and ntaper > 0:
                curr_win[:ntaper] = 1.
            elif fr == nfr - 1 and ntaper > 0:
                curr_win[-ntaper:] = 1.
            y[fr*nshift:fr*nshift + nwin,:] += x_fr * curr_win[:,np.newaxis]
        return y


    def _read_speech_file(self, infile, do_sox=True):
        if do_sox:
            os.mkdir('tmp')
            tmpfile = "tmp/{}_tmp.wav".format(infile.split('/')[-1].split('.')[0])
            cmd = "sox -V1 --no-dither {} -b 16 {}".format(infile, tmpfile)
            os.system(cmd)
            fs_o, x = read_wavfile(tmpfile)
            os.remove(tmpfile)
            os.rmdir('tmp')
        else:
            fs_o, x = read_wavfile(infile)
        if fs_o != self._fs:
            os.mkdir('tmp')
            tmpfile = "tmp/{}_tmp.wav".format(infile.split('/')[-1].split('.')[0])
            cmd = "sox -V1 --no-dither {} -r {} -b 16 {}".format(infile, self._fs, tmpfile)
            os.system(cmd)
            fs, x = read_wavfile(tmpfile)
            os.remove(tmpfile)
            os.rmdir('tmp')
        x /= 2**15
        return x, fs_o


    def _get_audiogram(self, audiogram_file):
        audiogram = pd.read_csv(audiogram_file)
        freqs = audiogram['Frequency'].tolist()
        dBloss = audiogram['dB Loss'].tolist()
        nfreq = len(freqs)
        freqs2 = np.logspace(np.log10(125), np.log10(8000), self._out_channels)
        audiogram = np.interp(freqs2, freqs, dBloss)
        cohc, cihc, _ = fitaudiogram3(freqs=freqs2, dBLoss=audiogram)
        return cohc, cihc


    @staticmethod
    def serialize(x):
        if isinstance(x,float) and np.abs(x) < 1. and np.abs(x) > 0.0:
            return "{}{}".format(int(np.abs(x) * 10**(np.ceil(-np.log10(np.abs(x)))+2)),
                                 int(np.ceil(-np.log10(np.abs(x)))))
        elif isinstance(x,int) and np.abs(x) > 100.:
            return int(np.abs(x) * 10**(np.ceil(-np.log10(np.abs(x)))+2))
        elif isinstance(x,int):
            return int(np.abs(x))
        elif isinstance(x,bool):
            return int(x)
        elif isinstance(x,float):
            return int(10 * np.abs(x))
        else:
            return x
    

    def _create_modelname(self):
        if self._channel_depth == 160:
            self.model_name = os.path.join(self._model_dir, 'earnet_large.h5')
        elif self._channel_depth == 80:
            self.model_name = os.path.join(self._model_dir, 'earnet_small.h5')
        else:
            assert(0)
