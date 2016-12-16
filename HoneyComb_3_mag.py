#-*- coding: utf8 -*-
"""
Spheres on two hexagonal close packed layers
"""
import numpy
import matplotlib
import pylab
from numpy import sin, cos, arange, pi, linspace, array, ones, abs, meshgrid, savetxt
from bornagain import *

alpha_i_min, alpha_i_max=0.0, 4.0
alpha_i=0.35
lambda_i=5.5

phi_f_min, phi_f_max=-1.9, 1.75
alpha_f_min, alpha_f_max=0.0, 4.0
tth_min, tth_max=-0.08, 4.4

lattice_rotation=0.

def get_sample():
    """
    Build and return the sample representing spheres on two hexagonal close packed layers
    """
    film_thickness=13.*nanometer
    lattice_length=20.0*nanometer
    radius=lattice_length*0.6
    cylinder_ff=FormFactorCylinder(radius, film_thickness)
    spin_ff=FormFactorBox(lattice_length*0.8, 0.2*lattice_length, film_thickness)
    surface_fraction=pi*radius**2/(3.*lattice_length**2*sin(60.*degree))

    M1=kvector_t(3.*sin(lattice_rotation*degree), 3.*cos(lattice_rotation*degree), 0.)
    M2=kvector_t(3.*sin(lattice_rotation*degree), 3.*cos(lattice_rotation*degree), 0.)
    M3=kvector_t(3.*sin(lattice_rotation*degree), 3.*cos(lattice_rotation*degree), 0.)
    m_air=HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=HomogeneousMaterial("Silicon", 2.07261e-06, 0.)
    m_particle=HomogeneousMagneticMaterial("PermalloyHole", (-1.-surface_fraction)*9.09538e-06, 0., 0.*M1)
    m_layer=HomogeneousMagneticMaterial("PermalloyLayer", (1.-surface_fraction)*9.09538e-06, 0., 0.*M1)
    m_mag_1p=HomogeneousMagneticMaterial("SpinTop", 0., 0., M1)
    m_mag_1m=HomogeneousMagneticMaterial("SpinBot", 0., 0.,-M1)
    m_mag_2p=HomogeneousMagneticMaterial("SpinTopL", 0., 0., M2)
    m_mag_2m=HomogeneousMagneticMaterial("SpinBotR", 0., 0.,-M2)
    m_mag_3p=HomogeneousMagneticMaterial("SpinTopR", 0., 0., M3)
    m_mag_3m=HomogeneousMagneticMaterial("SpinBotL", 0., 0.,-M3)

    #rotation_0=Transform3D.createRotateZ(lattice_rotation*degree)
    cylinder=Particle(m_particle, cylinder_ff)
    rotation=Transform3D.createRotateZ((lattice_rotation+90.)*degree)
    spin_p=Particle(m_mag_p, spin_ff, rotation)
    spin_m=Particle(m_mag_m, spin_ff, rotation)

    origin=kvector_t(0.0, 0.0, 0.0)
    basis_vec1=kvector_t(cos((lattice_rotation+30.)*degree)*lattice_length*1.73,
                         sin((lattice_rotation+30.)*degree)*lattice_length*1.73, 0.)
    basis_vec2=kvector_t(cos((lattice_rotation+150.)*degree)*lattice_length*1.73,
                         sin((lattice_rotation+150.)*degree)*lattice_length*1.73, 0.)
    # nuclear lattice is continous layer minus cyllinders, 3 Honey combs per magnetic UC
    basis=LatticeBasis()
    basis.addParticle(cylinder, [origin, basis_vec1, basis_vec2])
    # magnetic lattice
    basis.addParticle(spin_p, [basis_vec1/2., basis_vec2+basis_vec1/2.])
    basis.addParticle(spin_m, [-basis_vec1/2., basis_vec2-basis_vec1/2.])
    particle_layout=ParticleLayout()
    particle_layout.addParticle(basis)

    interference=InterferenceFunction2DLattice.createHexagonal(2.*lattice_length,
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
    phi_distr=DistributionGaussian(0.0*degree, 0.05*degree)
    #simulation.addParameterDistribution("*/Beam/wavelength", wavelength_distr, 5)
    #simulation.addParameterDistribution("*/Beam/alpha", alpha_distr, 5)
    simulation.addParameterDistribution("*/Beam/phi", phi_distr, 9)
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

    lambda_i=4.7

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    uu=simulation.getPolarizedIntensityData(0, 0).getArray()
    dd=simulation.getPolarizedIntensityData(1, 1).getArray()
    ud=simulation.getPolarizedIntensityData(0, 1).getArray()
    for lattice_rotation in arange(15., 180., 15.):
      print lattice_rotation
      sample=get_sample()
      simulation=get_simulation()
      simulation.setSample(sample)
      simulation.runSimulation()
      uu+=simulation.getPolarizedIntensityData(0, 0).getArray()
      dd+=simulation.getPolarizedIntensityData(1, 1).getArray()
      ud+=simulation.getPolarizedIntensityData(0, 1).getArray()

    # showing the result
    im=pylab.imshow(numpy.rot90(uu+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-up'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
    pylab.figure()
    im=pylab.imshow(numpy.rot90(dd+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ down-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
    pylab.figure()
    im=pylab.imshow(numpy.rot90(ud+1., 1), norm=matplotlib.colors.LogNorm(1e2, 1e8),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

#    f=open('HoneyCombSimulation_pGISANS_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
#    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
#    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
#    Qz, Qy=meshgrid(qz, qy)
#    for i in range(248):
#      savetxt(f, array([Qy[i], Qz[i], result[i]]).T)
#      f.write('\n')
#    f.close()

#    lambda_i=5.4
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
#                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
#                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
#                    aspect='auto', cmap='gist_ncar')
#    cb=pylab.colorbar(im)
#    pylab.title('$\lambda=%.2f$'%lambda_i)
#    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
#    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
#    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
#
#    f=open('HoneyCombSimulation_pGISANS_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
#    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
#    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
#    Qz, Qy=meshgrid(qz, qy)
#    for i in range(248):
#      savetxt(f, array([Qy[i], Qz[i], result[i]]).T)
#      f.write('\n')
#    f.close()
#
#    lambda_i=6.1
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
#                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
#                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
#                    aspect='auto', cmap='gist_ncar')
#    cb=pylab.colorbar(im)
#    pylab.title('$\lambda=%.2f$'%lambda_i)
#    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
#    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
#    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
#
#    f=open('HoneyCombSimulation_pGISANS_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
#    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
#    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
#    Qz, Qy=meshgrid(qz, qy)
#    for i in range(248):
#      savetxt(f, array([Qy[i], Qz[i], result[i]]).T)
#      f.write('\n')
#    f.close()
#
#    lambda_i=6.8
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
#                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
#                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
#                    aspect='auto', cmap='gist_ncar')
#    cb=pylab.colorbar(im)
#    pylab.title('$\lambda=%.2f$'%lambda_i)
#    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
#    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
#    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
#
#    f=open('HoneyCombSimulation_GISANS_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
#    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
#    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
#    Qz, Qy=meshgrid(qz, qy)
#    for i in range(248):
#      savetxt(f, array([Qy[i], Qz[i], result[i]]).T)
#      f.write('\n')
#    f.close()


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

 
