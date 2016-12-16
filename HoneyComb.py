#-*- coding: utf8 -*-
"""
Spheres on two hexagonal close packed layers
"""
import numpy
import matplotlib
import pylab
from numpy import sin, cos, arange, pi, linspace, array, ones, abs
from bornagain import *

alpha_i_min, alpha_i_max=0.0, 4.0
alpha_i=0.3
lambda_i=5.5

phi_f_min, phi_f_max=-2.0, 1.65
alpha_f_min, alpha_f_max=0.0, 4.0
tth_min, tth_max=-0.48, 4.0

lattice_rotation=0.

def get_sample():
    """
    Build and return the sample representing spheres on two hexagonal close packed layers
    """
    m_air=HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=HomogeneousMaterial("Silicon", 2.07261e-06, 0.)
    m_particle=HomogeneousMaterial("Permalloy", 1.33*9.09538e-06, 0.)
    m_particle_p=HomogeneousMaterial("Permalloy+", 1.33*9.09538e-06+3e-6, 0.)
    m_particle_m=HomogeneousMaterial("Permalloy-", 1.33*9.09538e-06-3e-6, 0.)
    m_layer=HomogeneousMaterial("PermalloyLayer", 0.33*9.09538e-06, 0.)

    film_thickness=13.*nanometer
    lattice_length=20.0*nanometer
    prism_ff=FormFactorPrism3(lattice_length, film_thickness)

    rotation_0=Transform3D.createRotateZ(lattice_rotation*degree)
    prism=Particle(m_particle, prism_ff, rotation_0)
    rotation_1=Transform3D.createRotateZ((lattice_rotation+180.)*degree)
    prism_rot=Particle(m_particle, prism_ff, rotation_1)

    pos0=kvector_t(0.0, 0.0, 0.0)
    pos1=kvector_t(-sin(lattice_rotation*degree)*lattice_length,
                   cos(lattice_rotation*degree)*lattice_length, 0.)
    basis=LatticeBasis()
    basis.addParticle(prism, [pos0])
    basis.addParticle(prism_rot, [pos1])
    particle_layout=ParticleLayout()
    particle_layout.addParticle(basis)

    interference=InterferenceFunction2DLattice.createHexagonal(lattice_length*2.0,
                                                               lattice_rotation*degree)
    pdf=FTDistribution2DCauchy(150*nanometer, 150*nanometer)
    interference.setProbabilityDistribution(pdf)

    particle_layout.addInterferenceFunction(interference)

    multi_layer=MultiLayer()
    air_layer=Layer(m_air)
    multi_layer.addLayer(air_layer)

    part_layer=Layer(m_layer, film_thickness)
    part_layer.addLayout(particle_layout)
    multi_layer.addLayer(part_layer)

    substrate_layer=Layer(m_substrate, 0)
    multi_layer.addLayer(substrate_layer)
    return multi_layer


def get_simulation():
    """
    Create and return GISAXS simulation with beam and detector defined
    """
    simulation=Simulation()
    simulation.setDetectorParameters(248, phi_f_min*degree, phi_f_max*degree, 296,
                                     (tth_min-alpha_i)*degree, (tth_max-alpha_i)*degree)
    simulation.setBeamParameters(lambda_i*angstrom, alpha_i*degree, 0.0*degree)

    #wavelength_distr=DistributionGaussian(lambda_i*angstrom, 0.1)
    #alpha_distr=DistributionGaussian(alpha_i*degree, 0.1*degree)
    #phi_distr=DistributionGaussian(0.0*degree, 0.1*degree)
    #simulation.addParameterDistribution("*/Beam/wavelength", wavelength_distr, 5)
    #simulation.addParameterDistribution("*/Beam/alpha", alpha_distr, 5)
    #simulation.addParameterDistribution("*/Beam/phi", phi_distr, 5)
    return simulation

def get_simulation_offspec():
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=OffSpecSimulation()
    simulation.setDetectorParameters(10, phi_f_min*degree, phi_f_max*degree, 296,
                                     alpha_f_min*degree, alpha_f_max*degree)
    # defining the beam  with incidence alpha_i varied between alpha_i_min and alpha_i_max
    alpha_i_axis=FixedBinAxis("alpha_i", 1, alpha_i*degree, alpha_i*degree)
    simulation.setBeamParameters(lambda_i*angstrom, alpha_i_axis, 0.0*degree)
    simulation.setBeamIntensity(1e6*alpha_i**2)
    return simulation

def get_simulation_spec():
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=SpecularSimulation()
    simulation.setBeamParameters(4.*pi*angstrom, 1000, 0., 0.3)
    return simulation

def run_simulation():
    """
    Run simulation and plot results
    """
    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    result=simulation.getIntensityData().getArray()+1  # for log scale

    # showing the result
    im=pylab.imshow(numpy.rot90(result, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[phi_f_min, phi_f_max, alpha_f_min, alpha_f_max], aspect='auto')
    cb=pylab.colorbar(im)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$\phi_f (^{\circ})$', fontsize=16)
    pylab.ylabel(r'$\alpha_f (^{\circ})$', fontsize=16)
    pylab.show()

def run_simulation_powder():
    """
    Run simulation and plot results
    """
    global lattice_rotation, lambda_i

    lambda_i=4.3

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    result=simulation.getIntensityData().getArray()
    for lattice_rotation in arange(5., 180., 5.):
      print lattice_rotation
      sample=get_sample()
      simulation=get_simulation()
      simulation.setSample(sample)
      simulation.runSimulation()
      result+=simulation.getIntensityData().getArray()

    # showing the result
    im=pylab.imshow(numpy.rot90(result+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
                 extent=[4*pi/lambda_i*phi_f_min*degree, 4*pi/lambda_i*phi_f_max*degree,
                         4*pi/lambda_i*tth_min*degree, 4*pi/lambda_i*tth_max*degree], aspect='auto')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

#    lambda_i=5.8
#
#    sample=get_sample()
#    simulation=get_simulation()
#    simulation.setSample(sample)
#    simulation.runSimulation()
#    result=simulation.getIntensityData().getArray()
#    for lattice_rotation in arange(5., 180., 5.):
#      print lattice_rotation
#      sample=get_sample()
#      simulation=get_simulation()
#      simulation.setSample(sample)
#      simulation.runSimulation()
#      result+=simulation.getIntensityData().getArray()
#
#    # showing the result
#    pylab.figure()
#    im=pylab.imshow(numpy.rot90(result+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
#                 extent=[4*pi/lambda_i*phi_f_min*degree, 4*pi/lambda_i*phi_f_max*degree,
#                         4*pi/lambda_i*tth_min*degree, 4*pi/lambda_i*tth_max*degree], aspect='auto')
#    cb=pylab.colorbar(im)
#    pylab.title('$\lambda=%.2f$'%lambda_i)
#    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
#    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
#    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
#
#    lambda_i=7.1
#
#    sample=get_sample()
#    simulation=get_simulation()
#    simulation.setSample(sample)
#    simulation.runSimulation()
#    result=simulation.getIntensityData().getArray()
#    for lattice_rotation in arange(5., 180., 5.):
#      print lattice_rotation
#      sample=get_sample()
#      simulation=get_simulation()
#      simulation.setSample(sample)
#      simulation.runSimulation()
#      result+=simulation.getIntensityData().getArray()
#
#    # showing the result
#    pylab.figure()
#    im=pylab.imshow(numpy.rot90(result+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
#                 extent=[4*pi/lambda_i*phi_f_min*degree, 4*pi/lambda_i*phi_f_max*degree,
#                         4*pi/lambda_i*tth_min*degree, 4*pi/lambda_i*tth_max*degree], aspect='auto')
#    cb=pylab.colorbar(im)
#    pylab.title('$\lambda=%.2f$'%lambda_i)
#    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
#    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
#    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    pylab.show()


def run_offspec_powder():
    """
    Run simulation and plot results
    """
    global lattice_rotation, alpha_i, lambda_i
    lambda_i=5.8
    alpha_i=0.3

    sample=get_sample()
    result=[]
    pis=[]
    pfs=[]
    for lambda_i in linspace(7.8, 2.6, 60):
      print lambda_i
      pis.append(2.*pi/lambda_i*alpha_i*degree*ones(296))
      pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296)*degree)
      simulation=get_simulation_offspec()
      simulation.setSample(sample)
      simulation.runSimulation()
      result.append(simulation.getIntensityData().getArray()[0])

    for alpha_i in [0.54, 0.97, 1.75]:
      for lambda_i in linspace(5.4, 2.6, 40):
        print lambda_i
        pis.append(2.*pi/lambda_i*alpha_i*degree*ones(296))
        pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296)*degree)
        simulation=get_simulation_offspec()
        simulation.setSample(sample)
        simulation.runSimulation()
        result.append(simulation.getIntensityData().getArray()[0])

    result=array(result)
    pis=array(pis); pfs=array(pfs)
    print result.shape, pis.shape, pfs.shape

    pylab.figure()
    im=pylab.pcolormesh(pis-pfs, pis+pfs, result, norm=matplotlib.colors.LogNorm(1e4, 1e9))
    cb=pylab.colorbar(im)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$p_i-p_f (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    simulation=get_simulation_spec()
    simulation.setSample(sample)
    simulation.runSimulation()
    result=simulation.getScalarR(0)
    pylab.figure()
    pylab.semilogy(abs(result))

    pylab.show()


if __name__=='__main__':
    run_simulation_powder()
    #run_offspec_powder()

 
