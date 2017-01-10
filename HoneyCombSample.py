#-*- coding: utf-8 -*-
'''
Definition of the sample model for the Honeycomb lattice simulation.

Author: Artur Glavic

BornAgain version 1.7.1  
'''

import bornagain as ba
from ba_helper import MLBuilder
from numpy import pi, sin, cos, sqrt

# define some abbreviations
deg=ba.deg;AA=ba.angstrom;nm=ba.nm

# constants used for the model
SLD_Si=2.029e-6*AA**-2 #
SLD_Py=9.152e-6*AA**-2 # Fe0.2Ni0.8 with density 8.72 g/cm³
SLD_Top=2.5e-6*AA**-2 # Unknown surface layer
s3=sqrt(3.)


class HCSample(MLBuilder):
  """
  Implement a sample builder to allow coupling of certain parameters to each other.
  """
  # initialize parameters
  @property
  def py_d(self): return self._py_d.value
  @property
  def top_d(self): return self._top_d.value
  @property
  def hc_lattice_length(self): return self._hc_lattice_length.value
  @property
  def hc_inner_radius(self): return self._hc_inner_radius.value
  @property
  def xi(self): return self._xi.value
  mag_lattice_length=None
  bias_field=0.
  edge_field=1.0
  lambda_i=None
  domain=None
  cauchy=None
  interference_model='paracrystal'

  def __init__(self, py_d=13.4*nm, top_d=8.0*nm,
               hc_lattice_length=35.0*nm, hc_inner_radius=13.0*nm,
               magnetic_model='vortex', bias_field=0.0, edge_field=1.0,
               xi=0., domain=250.*nm, cauchy=15*nm,
               lambda_i=4.0):
    ba.IMultiLayerBuilder.__init__(self)

    # global parameters, ctypes first:
    # Thickness of Permalloy and surface layer
    self._py_d=self._registerParameter('py_d', py_d)
    self._top_d=self._registerParameter('top_d', top_d)
    # Distance between two honeycomb cells
    self._hc_lattice_length=self._registerParameter('hc_lattice_length', hc_lattice_length)
    # radius of the cylinder cut out that produces the honeycomb structure
    self._hc_inner_radius=self._registerParameter('hc_inner_radius', hc_inner_radius)
    # rotation of the lattice to the incident beam
    self._xi=self._registerParameter('xi', xi)
    # normal python attributes
    self.magneti_model=magnetic_model
    self.bias_field=bias_field
    self.edge_field=edge_field
    self.lambda_i=lambda_i
    self.domain=domain
    self.cauchy=cauchy

  def buildSample(self):
    '''
    Constructs the sample for current values of parameters.
    '''
    sf=(self.lambda_i*AA)**2/2./pi
    #print self.lambda_i, SLD_Py*sf

    # the magnetic unti cell lattice length, 30degree tilted, sqrt(3)xsqrt(3)
    self.mag_lattice_length=self.hc_lattice_length*s3
    # calculate the surface filling density from the lattice size and cylinder
    self.surface_fraction=1.-pi*self.hc_inner_radius**2/(self.hc_lattice_length**2*sin(60.*deg))

    # external field direction
    B_ext=ba.kvector_t(0., 1., 0.)

    # define materials used in the model
    m_air=ba.HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=ba.HomogeneousMaterial("Silicon", SLD_Si*sf, 0.)
    # average density of the Py layer
    m_layer=ba.HomogeneousMagneticMaterial("PermalloyLayer", self.surface_fraction*SLD_Py*sf, 0.,
                                           self.surface_fraction*B_ext*self.bias_field*sf)
    m_top=ba.HomogeneousMaterial("SurfaceLayer", self.surface_fraction*SLD_Top*sf, 0.)

    # initialize model and ambiance layer
    multi_layer=ba.MultiLayer()
    air_layer=ba.Layer(m_air)
    multi_layer.addLayer(air_layer)

    #roughness=ba.LayerRoughness()
    #roughness.setSigma(200.0*nm)
    #roughness.setHurstParameter(0.7)
    #roughness.setLatteralCorrLength(2500.0*nm)

    top_layer=ba.Layer(m_top, self.top_d)
    #multi_layer.addLayerWithTopRoughness(top_layer, roughness)
    multi_layer.addLayer(top_layer)

    # generate honeycomb lattice and add it's layer to the model
    particle_layout=self.build_lattice()
    py_layer=ba.Layer(m_layer, self.py_d)
    py_layer.addLayout(particle_layout)
    #multi_layer.addLayerWithTopRoughness(py_layer, roughness)
    multi_layer.addLayer(py_layer)
    substrate_layer=ba.Layer(m_substrate, 0)
    #multi_layer.addLayerWithTopRoughness(substrate_layer, roughness)
    multi_layer.addLayer(substrate_layer)

    #multi_layer.setCrossCorrLength(250*nm)
    return multi_layer


  def build_lattice(self):
    '''
    Generate a lattice of negative cylinders to cut-out of the full layer to form the
    honeycomb structure.
    Add magnetic parts to the lattice and return the particle layout of it.
    '''
    sf=(self.lambda_i*AA)**2/2./pi

    # particle SLD to produce a contrast between average density and particle of Py-SLD
    # external field direction
    B_ext=ba.kvector_t(0., 1., 0.)
    m_hole=ba.HomogeneousMagneticMaterial("PermalloyHole", 0., 0.,
                                          0.*B_ext)
    m_full=ba.HomogeneousMagneticMaterial("PermalloyFull", SLD_Py*sf, 0.,
                                          (self.surface_fraction-1.)*B_ext*self.bias_field*sf)
    ll=self.hc_lattice_length
    mll=self.mag_lattice_length

    cylinder_ff=ba.FormFactorCylinder(self.hc_inner_radius, self.py_d)
    cylinder=ba.Particle(m_hole, cylinder_ff)
    hexagon_ff=ba.FormFactorPrism6(mll/2., self.py_d)

    o=ba.kvector_t(0.0, 0.0,-self.py_d) # origin is the bottom of the py layer
    basis=ba.ParticleComposition()
    particle_layout=ba.ParticleLayout()
    
    if self.magneti_model=='ferro':
      hexagon=ba.Particle(m_full, hexagon_ff)
      cell=ba.ParticleCoreShell(hexagon, cylinder)
      # for pure ferromagnetism the unit cell can be reduced to one particle.
      basis.addParticles(cell, [o]) # nuclear unit cell
      particle_layout.addParticle(basis, 1.0, ba.kvector_t(0, 0, 0), ba.RotationZ((self.xi-30.)*deg))
    else:
      hexagon=ba.Particle(m_full, hexagon_ff)
      cell=ba.ParticleCoreShell(hexagon, cylinder)
      # vortex magnetic model
      ln_a=ba.kvector_t(cos(30.*deg)*ll, sin(30.*deg)*ll, 0.)
      ln_b=ba.kvector_t(cos(90.*deg)*ll, sin(90.*deg)*ll, 0.)

      basis.addParticles(cell, [o, o+ln_a, o+ln_b ]) # nuclear unit cell hole
      self.build_votex(basis) # magnetic unit cell
      particle_layout.addParticle(basis, 1.0, ba.kvector_t(0, 0, 0), ba.RotationZ((self.xi-30.)*deg))

    interference=self.get_interference_function()
    particle_layout.addInterferenceFunction(interference)
    return particle_layout

  def get_interference_function(self):

    if self.magneti_model=='ferro':
      ll=self.hc_lattice_length
      rot=self.xi
    else:
      ll=self.mag_lattice_length
      rot=self.xi-30.

    if self.interference_model=='crystal':
      interference=ba.InterferenceFunction2DLattice(ll, ll, 60.*deg, rot*deg)
      pdf=ba.FTDecayFunction2DCauchy(self.cauchy, self.cauchy)
      interference.setDecayFunction(pdf)
    else:
      interference=ba.InterferenceFunction2DParaCrystal(ll, ll, 60.*deg, rot*deg, self.domain)
      interference.setDomainSizes(self.domain, self.domain)
      pdf=ba.FTDistribution2DCauchy(self.cauchy, self.cauchy)
      interference.setProbabilityDistributions(pdf, pdf)
      interference.setIntegrationOverXi(False)
    return interference

  def build_votex(self, basis):
    '''
    Add magnetic particles to the basis used in build_lattice conform to a vortex model.
    '''
    sf=self.lambda_i**2/2./pi
    ll=self.hc_lattice_length
    mll=self.mag_lattice_length

    spin_ff=ba.FormFactorBox(ll*0.4, 0.1*ll, self.py_d)

    M=ba.kvector_t(cos(self.xi*deg), sin(self.xi*deg), 0.)*self.edge_field
    m_mag_000=ba.HomogeneousMagneticMaterial("Spin", 0., 0., M*sf)
    m_mag_180=m_mag_000
    m_mag_030=m_mag_000
    m_mag_240=m_mag_000
    m_mag_330=m_mag_000
    m_mag_120=m_mag_000

    rotation=ba.RotationZ(0.*deg)
    spin_1p=ba.Particle(m_mag_000, spin_ff, rotation)
    rotation=ba.RotationZ(180.*deg)
    spin_1m=ba.Particle(m_mag_180, spin_ff, rotation)

    rotation=ba.RotationZ(30.*deg)
    spin_2p=ba.Particle(m_mag_030, spin_ff, rotation)
    rotation=ba.RotationZ(240.*deg)
    spin_2m=ba.Particle(m_mag_240, spin_ff, rotation)

    rotation=ba.RotationZ(330.*deg)
    spin_3p=ba.Particle(m_mag_330, spin_ff, rotation)
    rotation=ba.RotationZ(120.*deg)
    spin_3m=ba.Particle(m_mag_120, spin_ff, rotation)

    o=ba.kvector_t(0.0, 0.0,-self.py_d) # origin is the bottom of the py layer
    ln_a=ba.kvector_t(cos(30.*deg)*ll, sin(30.*deg)*ll, 0.)
    ln_b=ba.kvector_t(cos(90.*deg)*ll, sin(90.*deg)*ll, 0.)
    lm_a=ba.kvector_t(mll, 0., 0.)
    lm_b=ba.kvector_t(cos(60.*deg)*mll, sin(60.*deg)*mll, 0.)
    om=o+lm_a/2.+ba.kvector_t(0., 0., self.py_d/6.) # magnetic lattice origin is in the middle between two cylinders
    basis.addParticles(spin_1p, [om])
    basis.addParticles(spin_1m, [om+ln_b, om-ln_b])

    basis.addParticles(spin_2p, [om+ln_b/2.+lm_a/4., om+ln_b*3./2.+lm_a/4.])
    basis.addParticles(spin_2m, [o+lm_b/2.])

    basis.addParticles(spin_3p, [o+ln_a/2., o+ln_a*5./2.])
    basis.addParticles(spin_3m, [o+ln_a*3./2.])

if __name__=='__main__':
  print "This is just a module containing the sample description, run HoneyComb.py to simulate."
