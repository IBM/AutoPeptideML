import json

import numpy as np

from autopeptideml.reps.lms import RepEngineLM
from autopeptideml.reps.seq_based import RepEngineOnehot
from autopeptideml.reps.fps import RepEngineFP


def test_esm_family():
    re = RepEngineLM('esm2-8m', average_pooling=True)
    a = re.compute_reps(['AACFFF'], batch_size=12)
    b = re.compute_reps(['AACFFF', 'AACCF'], batch_size=12)
    re = RepEngineLM('esm2-8m', average_pooling=False)
    c = re.compute_reps(['AACFFF'], batch_size=12)

    assert re.dim() == 320
    assert a.shape == (1, 320)
    assert b.shape == (2, 320)
    assert np.array(c).shape == (1, 6, 320)


def test_elnaggar_family():
    re = RepEngineLM('ankh-base')
    a = re.compute_reps(['AACFFF'], batch_size=12)
    assert re.dim() == 768
    assert np.array(a).shape == (1, re.dim())


def test_one_hot():
    re = RepEngineOnehot(19)
    a = re.compute_reps(['AACFFF', 'AACCF'], batch_size=4)
    dict_re = json.loads(str(re))
    assert dict_re == {'rep': 'one-hot', 'max_length': 19}
    assert re.dim() == 19 * 21
    assert a.shape == (2, 19 * 21)


def test_fps():
    re1 = RepEngineFP('ecfp', 512, 8)
    a = re1.compute_reps(['C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]'], batch_size=1)
    re2 = RepEngineFP('fcfp', 256, 12)
    b = re2.compute_reps(['C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@H]'], batch_size=1)
    dict_1, dict_2 = json.loads(str(re1)), json.loads(str(re2))
    assert dict_1 == {'rep': 'ecfp', 'nbits': 512, 'radius': 8}
    assert dict_2 == {'rep': 'fcfp', 'nbits': 256, 'radius': 12}
    assert re1.dim() == 512
    assert re2.dim() == 256
    assert a.shape == (1, 512)
    assert b.shape == (1, 256)


def test_rostlab_family():
    re = RepEngineLM('prot-t5-xl')
    a = re.compute_reps(['AACFFF'], batch_size=12)
    assert re.dim() == 1024
    assert np.array(a).shape == (1, re.dim())


if __name__ == '__main__':
    test_esm_family()
    print('ESM OK')
    test_elnaggar_family()
    print('Elnaggar OK')
    test_one_hot()
    print('Onehot OK')
    test_fps()
    print("FPs OK")
    test_rostlab_family()
    print("Rostlab OK")
