"""Top-level package for AutoPeptideML."""

__author__ = """Raul Fernandez-Diaz"""
__email__ = 'raul.fernandezdiaz@ucdconnect.ie'
__all__ = ['AutoPeptideML', '__version__',
           'element_registry']

from .autopeptideml import AutoPeptideML, __version__
from .reps import *
from .preprocess import *
