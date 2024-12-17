from autopeptideml import RepresentationEngine
import numpy as np


def test_esm_family():
    re = RepresentationEngine('esm2-8m', batch_size=12)
    a = re.compute_representations(['AACFFF'], average_pooling=True)
    b = re.compute_representations(['AACFFF', 'AACCF'], average_pooling=True)
    c = re.compute_representations(['AACFFF'], average_pooling=False)
    assert re.dim() == 320
    assert np.array(a).shape == (1, 320)
    assert np.array(b).shape == (2, 320)
    assert np.array(c).shape == (1, 6, 320)


def test_elnaggar_family():
    re = RepresentationEngine('ankh-base', batch_size=12)
    a = re.compute_representations(['AACFFF'], average_pooling=True)
    assert re.dim() == 768
    assert np.array(a).shape == (1, re.dim())


def test_rostlab_family():
    re = RepresentationEngine('prot-t5-xl', batch_size=12)
    a = re.compute_representations(['AACFFF'], average_pooling=True)
    assert re.dim() == 1024
    assert np.array(a).shape == (1, re.dim())
