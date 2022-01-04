import numpy as np
import audio_features as af

from typing import List, Tuple
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class BaseTransformer(BaseEstimator, TransformerMixin):

    features_list: list
    verbose: bool = True

    all_functions = {
        'preprocess_audio': af.preprocess_audio, 'get_energy': af.get_energy,
        'get_energy_entropy': af.get_energy_entropy, 'get_spectral_entropy': af.get_spectral_entropy,
        'get_spectral_flux': af.get_spectral_flux, 'get_spectral_centroid': af.get_spectral_centroid,
        'get_spectral_spread': af.get_spectral_spread, 'get_spectral_contrast': af.get_spectral_contrast,
        'get_spectral_flatness': af.get_spectral_flatness, 'get_spectral_rolloff': af.get_spectral_rolloff,
        'get_zero_crossing_rate': af.get_zero_crossing_rate, 'get_harmonic_ratio': af.get_harmonic_ratio,
        'get_mfccs': af.get_mfccs,
        'get_rms': af.get_rms, 'get_poly_features': af.get_poly_features  # NEW FEATURE FUNCTIONS
    }
    all_func_params = {
        'preprocess_audio':       ['thr'],  # Preprocessing Trim + :enter
        'get_energy':             ['flen', 'hop'],
        'get_energy_entropy':     ['flen', 'hop', 'nsub'],
        'get_spectral_entropy':   ['flen', 'hop', 'nsub'],
        'get_spectral_flux':      ['flen', 'hop'],
        'get_spectral_centroid':  ['flen', 'hop', 'sr'],
        'get_spectral_spread':    ['flen', 'hop', 'porder', 'sr'],
        'get_spectral_contrast':  ['flen'],
        'get_spectral_flatness':  ['flen'],
        'get_spectral_rolloff':   ['flen', 'hop', 'perc', 'sr'],
        'get_zero_crossing_rate': ['flen', 'hop'],
        'get_harmonic_ratio':     ['flen', 'hop', 'sr'],
        'get_mfccs':              ['flen', 'hop', 'n_mfcc', 'sr'],
        'get_rms':                ['flen', 'hop'],
        'get_poly_features':      ['porder']
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        num_data = len(X)
        audio_sr = 16000  # 16000 for all the audios
        n_feat = len(self.features_list)

        audio_funcs = self.gen_audio_funcs()
        preprocess_func, pp_params = self.get_preprocess_function(audio_funcs)
        audio_functions = self.get_features_functions(audio_funcs)

        M = np.zeros((num_data, n_feat * len(audio_functions)))  # Generate empty feature matrix

        if self.verbose:
            self.print_results(audio_functions)

        for i in range(num_data):

            audio_data = preprocess_func(X[i], **pp_params)

            for idx, (audio_func, kwargs) in enumerate(audio_functions):
                kwargs = self.parse_kwargs(kwargs, audio_sr=audio_sr)
                features = audio_func(audio_data, **kwargs)

                for f_idx, feat_func in enumerate(self.features_list):
                    M[i, idx * n_feat + f_idx] = feat_func(features)

        return M

    def gen_audio_funcs(self) -> List[Tuple[callable, dict]]:
        audio_functions = []
        for name, func in self.all_functions.items():
            if getattr(self, name, False):
                audio_functions.append((func, {}))
        return audio_functions

    def get_preprocess_function(self, audio_funcs: list) -> (callable, dict):
        for func, params in audio_funcs:
            if func.__name__ == 'preprocess_data':
                return func, {p: getattr(self, p, True)
                              for p in params}
        return af.preprocess_audio, {'thr': 20}

    def get_features_functions(self, audio_funcs: list) -> List[Tuple[callable, dict]]:
        ffuncs = []
        for func, params in audio_funcs:
            if func.__name__ != 'preprocess_data':
                ffuncs.append((func, {p: getattr(self, p, True)
                                      for p in params}))
        return ffuncs

    @staticmethod
    def parse_kwargs(kwargs, audio_sr):
        if kwargs.get('sr'):
            kwargs['sr'] = audio_sr
        return kwargs

    @staticmethod
    def print_results(functions):
        print([func.__name__ for func, _ in functions])


@dataclass
class CombineFeatures(BaseTransformer):

    preprocess_audio:       bool = True
    get_energy:             bool = False
    get_energy_entropy:     bool = False
    get_spectral_entropy:   bool = False
    get_spectral_flux:      bool = False
    get_spectral_centroid:  bool = False
    get_spectral_spread:    bool = False
    get_spectral_contrast:  bool = False
    get_spectral_flatness:  bool = False
    get_spectral_rolloff:   bool = False
    get_zero_crossing_rate: bool = False
    get_harmonic_ratio:     bool = False
    get_mfccs:              bool = False
    get_rms:                bool = False
    get_poly_features:      bool = False


@dataclass
class ExtractFeatures(CombineFeatures):

    thr:    int = 20
    flen:   int = 1024
    hop:    int = 512
    nsub:   int = 10
    porder: int = 2
    perc:   float = 0.85
    n_mfcc: int = 20
    sr:     int = 16000

    def gen_audio_funcs(self) -> List[Tuple[callable, dict]]:
        audio_functions = []
        for name, func in self.all_functions.items():
            ff_params = {p: getattr(self, p, True) for p in self.all_func_params[name]}
            if getattr(self, name, False):
                audio_functions.append((func, ff_params))
        return audio_functions

    @staticmethod
    def print_results(functions):
        print([(func.__name__, params) for func, params in functions])
