#-*- coding: utf-8 -*-
'''
Some helper functions and classes to improve usability of BornAgain python API.


Author: Artur Glavic

BornAgain version 1.7.1
'''

import ctypes
import bornagain as ba

# define some abbreviations
deg=ba.deg;AA=ba.angstrom;nm=ba.nm

#expand kvector_t
ba.kvector_t.__sub__=lambda self, other: self+(-1.0*other)
ba.kvector_t.__div__=lambda self, other: self*(1./other)
ba.kvector_t.__neg__=lambda self: (-1.0)*self

class MLBuilder(ba.IMultiLayerBuilder):
  """
  Extension of builder class with parameter support method.
  """

  def _registerParameter(self, name, value):
    # convenience method to add parameters to the model
    param=ctypes.c_double(value)
    self.registerParameter(name, ctypes.addressof(param))
    return param


