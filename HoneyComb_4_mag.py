#-*- coding: utf8 -*-
"""
Spheres on two hexagonal close packed layers
"""
import numpy
import matplotlib
import pylab
from numpy import sin, cos, arange, pi, linspace, array, ones, abs, meshgrid, savetxt, sqrt
from bornagain import *
import bornagain as ba

alpha_i_min, alpha_i_max=0.0, 4.0
alpha_i=0.35
lambda_i=5.5

phi_f_min, phi_f_max=-1.9, 1.75
alpha_f_min, alpha_f_max=0.0, 4.0
tth_min, tth_max=-0.08, 4.4

lattice_rotation=0.

s3=sqrt(3.)

#expand kvector_t
kvector_t.__sub__=lambda self, other: self+(-1.0*other)
kvector_t.__div__=lambda self, other: self*(1./other)
kvector_t.__neg__=lambda self: (-1.0)*self

def get_sample():
    """
    Build and return the sample representing spheres on two hexagonal close packed layers
    """
    film_thickness=13.*nanometer
    lattice_length=20.0*nanometer
    radius=lattice_length*0.6
    cylinder_ff=FormFactorCylinder(radius, film_thickness)
    spin_ff=FormFactorBox(lattice_length*0.8, 0.2*lattice_length, film_thickness*2./3.)
    surface_fraction=pi*radius**2/(3.*lattice_length**2*sin(60.*degree))

    M1=kvector_t(3., 0., 0.)
    M2=kvector_t(3.*cos(30.*degree), 3.*sin(30.*degree), 0.)
    M3=kvector_t(3.*cos(-30.*degree), 3.*sin(-30.*degree), 0.)
    m_air=HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=HomogeneousMaterial("Silicon", 2.07261e-06, 0.)
    m_particle=HomogeneousMagneticMaterial("PermalloyHole", (-1.-surface_fraction)*9.09538e-06, 0., 0.*M1)
    m_layer=HomogeneousMagneticMaterial("PermalloyLayer", (1.-surface_fraction)*9.09538e-06, 0., 0.*M1)
    m_mag_1p=HomogeneousMagneticMaterial("SpinTop", 0., 0., M1*500.)
    m_mag_1m=HomogeneousMagneticMaterial("SpinBot", 0., 0., M1*500.)
    m_mag_2p=HomogeneousMagneticMaterial("SpinTopL", 0., 0., M1*500.)
    m_mag_2m=HomogeneousMagneticMaterial("SpinBotR", 0., 0., M1*500.)
    m_mag_3p=HomogeneousMagneticMaterial("SpinTopR", 0., 0., M1*500.)
    m_mag_3m=HomogeneousMagneticMaterial("SpinBotL", 0., 0., M1*500.)

    cylinder=Particle(m_particle, cylinder_ff)
    rotation=RotationZ((lattice_rotation+90.)*degree)
    spin_1p=Particle(m_mag_1p, spin_ff, rotation)
    spin_1m=Particle(m_mag_1m, spin_ff, rotation)
    spin_2p=Particle(m_mag_2p, spin_ff, rotation)
    spin_2m=Particle(m_mag_2m, spin_ff, rotation)
    spin_3p=Particle(m_mag_3p, spin_ff, rotation)
    spin_3m=Particle(m_mag_3m, spin_ff, rotation)

    origin=kvector_t(0.0, 0.0, 0.0)
    upmove=kvector_t(0., 0., film_thickness/6.)
    basis_vec1=kvector_t(cos((lattice_rotation+30.)*degree)*lattice_length*s3,
                         sin((lattice_rotation+30.)*degree)*lattice_length*s3, 0.)
    basis_vec2=kvector_t(cos((lattice_rotation+90.)*degree)*lattice_length*s3,
                         sin((lattice_rotation+90.)*degree)*lattice_length*s3, 0.)
    # nuclear lattice is continous layer minus cyllinders, 3 Honey combs per magnetic UC
    basis=ParticleComposition()
    basis.addParticles(cylinder, [origin, basis_vec1, basis_vec2])
    # magnetic lattice
    basis.addParticles(spin_1p, [basis_vec1-0.5*basis_vec2+upmove])
    basis.addParticles(spin_1m, [basis_vec1+basis_vec2/2.+upmove,
                                basis_vec1*2.-basis_vec2/2.+upmove])
    basis.addParticles(spin_2p, [basis_vec1*1.5-basis_vec2/2.+upmove,
                                basis_vec1*1.5+basis_vec2/2.+upmove])
    basis.addParticles(spin_2m, [basis_vec1/2.+basis_vec2/2.+upmove])
    basis.addParticles(spin_3p, [basis_vec1/2.+upmove,
                                basis_vec1*2.5+upmove])
    basis.addParticles(spin_3m, [basis_vec1*1.5+upmove])

    particle_layout=ParticleLayout()
    particle_layout.addParticle(basis)

    interference=InterferenceFunction2DParaCrystal.createHexagonal(lattice_length*3.0, 0.,
                                                               200.*nanometer, 200.*nanometer)
    pdf=FTDistribution2DCauchy(1*nanometer, 1*nanometer)
    interference.setProbabilityDistributions(pdf, pdf)

    particle_layout.addInterferenceFunction(interference)

    roughness=LayerRoughness()
    roughness.setSigma(1.0*nanometer)
    roughness.setHurstParameter(0.3)
    roughness.setLatteralCorrLength(100.0*nanometer)

    multi_layer=MultiLayer()
    air_layer=Layer(m_air)
    multi_layer.addLayer(air_layer)

    part_layer=Layer(m_layer, film_thickness)
    part_layer.addLayout(particle_layout)
    multi_layer.addLayerWithTopRoughness(part_layer, roughness)

    substrate_layer=Layer(m_substrate, 0)
    multi_layer.addLayerWithTopRoughness(substrate_layer, roughness)
    return multi_layer


def get_simulation():
    """
    Create and return GISAXS simulation with beam and detector defined
    """
    simulation=GISASSimulation()
    simulation.setDetectorParameters(248, phi_f_min*degree, phi_f_max*degree, 296,
                                     (tth_min-alpha_i)*degree, (tth_max-alpha_i)*degree)
    simulation.setBeamParameters(lambda_i*angstrom, alpha_i*degree, 0.0*degree)

    #wavelength_distr=DistributionGaussian(lambda_i*angstrom, 0.1)
    #alpha_distr=DistributionGaussian(alpha_i*degree, 0.1*degree)
    #phi_distr=DistributionGaussian(0.0*degree, 0.03*degree)
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
    up=kvector_t(1., 0., 0.)
    down=-up

    lambda_i=4.7

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(up, 1.0, 0.5)
    simulation.runSimulation()
    uu=simulation.getIntensityData().getArray()
    print 'uu'

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.setBeamPolarization(down)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    simulation.runSimulation()
    dd=simulation.getIntensityData().getArray()
    print 'dd'

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    ud=simulation.getIntensityData().getArray()
    print 'ud'
    print lambda_i

    # showing the result
    im=pylab.imshow(uu+0.01, norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-up'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(dd+0.01, norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ down-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(ud+0.00001, norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)
    pylab.show()

    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pp_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
    Qz, Qy=meshgrid(qz, qy)
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], uu[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_mm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], dd[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], ud[i]]).T)
      f.write('\n')
    f.close()


    lambda_i=5.4

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    uu=simulation.getPolarizedIntensityData(0, 0).getArray()
    dd=simulation.getPolarizedIntensityData(1, 1).getArray()
    ud=simulation.getPolarizedIntensityData(0, 1).getArray()
    print lambda_i

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(uu+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-up'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(dd+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ down-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(ud+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pp_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
    Qz, Qy=meshgrid(qz, qy)
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], uu[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_mm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], dd[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], ud[i]]).T)
      f.write('\n')
    f.close()

    lambda_i=6.1

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    uu=simulation.getPolarizedIntensityData(0, 0).getArray()
    dd=simulation.getPolarizedIntensityData(1, 1).getArray()
    ud=simulation.getPolarizedIntensityData(0, 1).getArray()
    print lambda_i

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(uu+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-up'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(dd+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ down-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(ud+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pp_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
    Qz, Qy=meshgrid(qz, qy)
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], uu[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_mm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], dd[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], ud[i]]).T)
      f.write('\n')
    f.close()

    lambda_i=6.8

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.runSimulation()
    uu=simulation.getPolarizedIntensityData(0, 0).getArray()
    dd=simulation.getPolarizedIntensityData(1, 1).getArray()
    ud=simulation.getPolarizedIntensityData(0, 1).getArray()
    print lambda_i

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(uu+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-up'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(dd+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ down-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    # showing the result
    pylab.figure()
    im=pylab.imshow(numpy.rot90(ud+0.01, 1), norm=matplotlib.colors.LogNorm(),
                 extent=[2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree,
                         2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.title('$\lambda=%.2f$ up-down'%lambda_i)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$Q_y (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pp_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    qy=linspace(2*pi/lambda_i*phi_f_min*degree, 2*pi/lambda_i*phi_f_max*degree, 248)
    qz=linspace(2*pi/lambda_i*tth_min*degree, 2*pi/lambda_i*tth_max*degree, 296)
    Qz, Qy=meshgrid(qz, qy)
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], uu[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_mm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], dd[i]]).T)
      f.write('\n')
    f.close()
    f=open('HoneyCombSimulation_ParacrystalMag_GISANS_pm_%.2fA_%.2fdegree.dat'%(lambda_i, alpha_i), 'w')
    for i in range(248):
      savetxt(f, array([Qy[i], Qz[i], ud[i]]).T)
      f.write('\n')
    f.close()



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

 
