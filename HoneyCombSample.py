#-*- coding: utf-8 -*-
'''
  doc
'''

import ctypes
import bornagain as ba
import pylab as plt

deg=ba.deg;AA=ba.angstrom;nm=ba.nm

class HCSample(ba.IMultiLayerBuilder):
  """
  """

  def _registerParameter(self, name, value):
    param=ctypes.c_double(value)
    self.registerParameter(name, ctypes.addressof(param))
    return param

  def __init__(self):
    ba.IMultiLayerBuilder.__init__(self)
    # parameters describing the sample
    self.length_parameter=ctypes.c_double(5.0*nm)
    self.decay_length=ctypes.c_double(10.0*nm)
    self.registerParameter("length_parameter", ctypes.addressof(self.length_parameter))
    self.registerParameter("decay_length", ctypes.addressof(self.decay_length))

  # constructs the sample for current values of parameters
  def buildSample(self):
    m_air=ba.HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=ba.HomogeneousMaterial("Substrate", 6e-6, 2e-8)
    m_particle=ba.HomogeneousMaterial("Particle", 6e-4, 2e-8)

    # correlation between sphere radius and lattice_length introduced: lattice_length is
    # always two times bigger than spheres radius
    radius=self.length_parameter.value
    lattice_length=2.0*self.length_parameter.value

    # same x,y decay length
    decay_length=self.decay_length.value

    sphere_ff=ba.FormFactorFullSphere(radius)
    sphere=ba.Particle(m_particle, sphere_ff)
    particle_layout=ba.ParticleLayout()
    particle_layout.addParticle(sphere)

    interference=ba.InterferenceFunction2DLattice.createHexagonal(lattice_length)
    pdf=ba.FTDecayFunction2DCauchy(decay_length, decay_length)
    interference.setDecayFunction(pdf)

    particle_layout.addInterferenceFunction(interference)

    air_layer=ba.Layer(m_air)
    air_layer.addLayout(particle_layout)
    substrate_layer=ba.Layer(m_substrate, 0)
    multi_layer=ba.MultiLayer()
    multi_layer.addLayer(air_layer)
    multi_layer.addLayer(substrate_layer)
    return multi_layer


def get_simulation():
  """
  Returns a GISAXS simulation with beam and detector defined.
  """
  simulation=ba.GISASSimulation()
  simulation.setDetectorParameters(100,-1.0*deg, 1.0*deg,
                                   100, 0.0*deg, 2.0*deg)
  simulation.setBeamParameters(1.0*AA, 0.2*deg, 0.0*deg)

  return simulation



if __name__=='__main__':
  sim=get_simulation()
  plt.show()
