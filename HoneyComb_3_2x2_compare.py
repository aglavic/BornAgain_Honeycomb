#-*- coding: utf8 -*-
"""
Spheres on two hexagonal close packed layers
"""
import numpy
import matplotlib
import pylab
from numpy import sin, cos, arange, pi, linspace, array, ones, abs, meshgrid, savetxt
from bornagain import *

alpha_i=0.35
lambda_i=5.5

phi_f_min, phi_f_max=-1.9, 1.75
tth_min, tth_max=-0.08, 4.4

lattice_rotation=0.

def get_sample(double_cell):
    """
    Build and return the sample representing spheres on two hexagonal close packed layers
    """
    film_thickness=13.*nanometer
    lattice_length=20.0*nanometer
    radius=lattice_length*0.6
    cylinder_ff=FormFactorCylinder(radius, film_thickness)
    surface_fraction=pi*radius**2/(3.*lattice_length**2*sin(60.*degree))

    m_air=HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=HomogeneousMaterial("Silicon", 2.07261e-06, 0.)
    m_particle=HomogeneousMaterial("PermalloyHole", (-1.-surface_fraction)*9.09538e-06, 0.)
    m_layer=HomogeneousMaterial("PermalloyLayer", (1.-surface_fraction)*9.09538e-06, 0.)

    cylinder=Particle(m_particle, cylinder_ff)

    origin=kvector_t(0.0, 0.0, 0.0)
    if double_cell:
      basis_vec1=kvector_t(cos(lattice_rotation*degree)*lattice_length*1.73,
                           sin(lattice_rotation*degree)*lattice_length*1.73, 0.)
      basis_vec2=kvector_t(cos((lattice_rotation+120.)*degree)*lattice_length*1.73,
                           sin((lattice_rotation+120.)*degree)*lattice_length*1.73, 0.)
      basis=LatticeBasis()
      basis.addParticle(cylinder, [origin, basis_vec1, basis_vec2, basis_vec1+basis_vec2])
      particle_layout=ParticleLayout()
      particle_layout.addParticle(basis)

      interference=InterferenceFunction2DLattice.createHexagonal(2.*lattice_length*1.73,
                                                                 lattice_rotation*degree)
    else:
      basis=LatticeBasis()
      basis.addParticle(cylinder, [origin])
      particle_layout=ParticleLayout()
      particle_layout.addParticle(basis)

      interference=InterferenceFunction2DLattice.createHexagonal(lattice_length*1.73,
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

    return simulation

def run_simulation_powder():
    """
    Run simulation and plot results
    """
    global lattice_rotation

    sample=get_sample(False)
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    result=simulation.getIntensityData().getArray()
#    for lattice_rotation in arange(5., 180., 5.):
#      print lattice_rotation
#      sample=get_sample(False)
#      simulation=get_simulation()
#      simulation.setSample(sample)
#      simulation.runSimulation()
#      result+=simulation.getIntensityData().getArray()

    # showing the result
    im=pylab.imshow(numpy.rot90(result+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('single particle cell')
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    sample=get_sample(True)
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    result=simulation.getIntensityData().getArray()
#    for lattice_rotation in arange(5., 180., 5.):
#      print lattice_rotation
#      sample=get_sample(True)
#      simulation=get_simulation()
#      simulation.setSample(sample)
#      simulation.runSimulation()
#      result+=simulation.getIntensityData().getArray()

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(result+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('2x2 particle cell')
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    pylab.show()


if __name__=='__main__':
    run_simulation_powder()

 
