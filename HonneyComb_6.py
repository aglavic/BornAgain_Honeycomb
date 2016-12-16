#-*- coding: utf-8 -*-
"""

"""
from numpy import sin, cos, arange, pi, linspace, array, ones, abs, meshgrid, savetxt, sqrt
from matplotlib.colors import LogNorm
import pylab
import bornagain as ba

A=ba.angstrom
nm=ba.nm
deg=ba.degree
Ms=0.
lattice_rotation=0.
is_FM=True

alpha_i_min, alpha_i_max=0.0, 4.0
alpha_i=0.35
lambda_i=5.5

phi_f_min, phi_f_max=-1.9, 1.75
alpha_f_min, alpha_f_max=0.0, 4.0
tth_min, tth_max=-0.08, 4.4

s3=sqrt(3.)

#expand kvector_t
ba.kvector_t.__sub__=lambda self, other: self+(-1.0*other)
ba.kvector_t.__div__=lambda self, other: self*(1./other)
ba.kvector_t.__neg__=lambda self: (-1.0)*self

def get_sample():
    """
    Build and return the sample representing spheres on two hexagonal close packed layers
    """
    Pt_film_thickness=7.6*nm
    Py_film_thickness=14.0*nm
    lattice_length=35.0*nm # closes distance between two honey comb centers
    mag_lattice_length=lattice_length*s3 # the magnetic unti cell lattice length, 30degree tilted, sqrt(3)xsqrt(3)
    radius=lattice_length*0.3

    cylinder_ff=ba.FormFactorCylinder(radius, Py_film_thickness)
    spin_ff=ba.FormFactorBox(lattice_length*0.4, 0.1*lattice_length, Py_film_thickness*2./3.)
    surface_fraction=pi*radius**2/(lattice_length**2*sin(60.*deg))

    m_air=ba.HomogeneousMaterial("Air", 0.0, 0.0)
    m_substrate=ba.HomogeneousMaterial("Silicon", 2.07261e-06, 0.)
    #m_particle2=ba.HomogeneousMaterial("PtHole", (-surface_fraction)*3.0e-06, 0.)
    m_layer2=ba.HomogeneousMaterial("PtLayer", (1.-surface_fraction)*3.0e-06, 0.)
    m_particle=ba.HomogeneousMaterial("PermalloyHole", (-surface_fraction)*9.09538e-06, 0.)
    m_layer=ba.HomogeneousMaterial("PermalloyLayer", (1.-surface_fraction)*9.09538e-06, 0.)

    if is_FM:
      m_mag_000=ba.HomogeneousMagneticMaterial("Spin1", 0., 0., ba.kvector_t(1., 0., 0.)*Ms)
      m_mag_180=ba.HomogeneousMagneticMaterial("Spin2", 0., 0., ba.kvector_t(-1., 0., 0.)*Ms)
      m_mag_030=ba.HomogeneousMagneticMaterial("Spin3", 0., 0.,
                                               ba.kvector_t(sin(-30*deg), cos(-30*deg), 0.)*Ms)
      m_mag_240=ba.HomogeneousMagneticMaterial("Spin4", 0., 0.,
                                               ba.kvector_t(sin(-30*deg), cos(-30*deg), 0.)*Ms)
      m_mag_330=ba.HomogeneousMagneticMaterial("Spin5", 0., 0., ba.kvector_t(1., 0., 0.)*Ms)
      m_mag_120=ba.HomogeneousMagneticMaterial("Spin6", 0., 0., ba.kvector_t(1., 0., 0.)*Ms)
    else:
      M=ba.kvector_t(1., 0., 0.)*Ms
      m_mag_000=ba.HomogeneousMagneticMaterial("Spin", 0., 0., M)
      m_mag_180=m_mag_000
      m_mag_030=m_mag_000
      m_mag_240=m_mag_000
      m_mag_330=m_mag_000
      m_mag_120=m_mag_000

    cylinder=ba.Particle(m_particle, cylinder_ff)

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

    o=ba.kvector_t(0.0, 0.0,-Py_film_thickness)
    ln_a=ba.kvector_t(cos(30.*deg)*lattice_length, sin(30.*deg)*lattice_length, 0.)
    ln_b=ba.kvector_t(cos(90.*deg)*lattice_length, sin(90.*deg)*lattice_length, 0.)

    lm_a=ba.kvector_t(mag_lattice_length, 0., 0.)
    lm_b=ba.kvector_t(cos(60.*deg)*mag_lattice_length, sin(60.*deg)*mag_lattice_length, 0.)
    om=o+lm_a/2.+ba.kvector_t(0., 0., Py_film_thickness/6.) # magnetic lattice origin is in the middle between two cylinders

    # nuclear lattice is continous layer minus cyllinders, 3 Honey combs per magnetic UC
    basis=ba.ParticleComposition()
    basis.addParticles(cylinder, [o, o+ln_a, o+ln_b ]) # nuclear unit cell
    # magnetic lattice
    basis.addParticles(spin_1p, [om])
    basis.addParticles(spin_1m, [om+ln_b, om-ln_b])

    basis.addParticles(spin_2p, [om+ln_b/2.+lm_a/4., om+ln_b*3./2.+lm_a/4.])
    basis.addParticles(spin_2m, [o+lm_b/2.])

    basis.addParticles(spin_3p, [o+ln_a/2., o+ln_a*5./2.])
    basis.addParticles(spin_3m, [o+ln_a*3./2.])

    part_layer=ba.Layer(m_layer, Py_film_thickness)

    for lattice_rotation in arange(0., 60., 15.0):
      particle_layout=ba.ParticleLayout()
      particle_layout.addParticle(basis, 1.0, ba.kvector_t(0, 0, 0), ba.RotationZ(lattice_rotation*deg))
      interference=ba.InterferenceFunction2DParaCrystal(mag_lattice_length, mag_lattice_length,
                                                        60.*deg,
                                                        lattice_rotation*deg, 0.*nm)
      interference.setDomainSizes(250*nm, 250*nm)
      pdf=ba.FTDistribution2DCauchy(7*nm, 7*nm)
      interference.setProbabilityDistributions(pdf, pdf)
      interference.setIntegrationOverXi(False)
      interference.printParameters()

      particle_layout.addInterferenceFunction(interference)

      part_layer.addLayout(particle_layout)

    roughness=ba.LayerRoughness()
    roughness.setSigma(14.0*nm)
    roughness.setHurstParameter(0.3)
    roughness.setLatteralCorrLength(250.0*nm)

    multi_layer=ba.MultiLayer()
    air_layer=ba.Layer(m_air)
    multi_layer.addLayer(air_layer)

    top_layer=ba.Layer(m_layer2, Pt_film_thickness)
#    multi_layer.addLayerWithTopRoughness(top_layer, roughness)
#    multi_layer.addLayerWithTopRoughness(part_layer, roughness)
    multi_layer.addLayer(top_layer)
    multi_layer.addLayer(part_layer)

    substrate_layer=ba.Layer(m_substrate, 0)
    #multi_layer.addLayerWithTopRoughness(substrate_layer, roughness)
    multi_layer.addLayer(substrate_layer)
    return multi_layer


# # Define functions for the simulation including plotting
# ## Simulation objec creation

# In[3]:

def get_simulation():
    """
    Create and return GISAXS simulation with beam and detector defined
    """
    simulation=ba.GISASSimulation()
    simulation.setDetectorParameters(248/2, phi_f_min*deg, phi_f_max*deg, 296/2,
                                     (tth_min-alpha_i)*deg, (tth_max-alpha_i)*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i*deg, 0.0*deg)
    simulation.setBeamIntensity(1e6)
    return simulation

def get_simulation_offspec():
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=ba.OffSpecSimulation()
    simulation.setDetectorParameters(11, phi_f_min*deg, phi_f_max*deg, 296/2,
                                     alpha_f_min*deg, alpha_f_max*deg)
    # defining the beam  with incidence alpha_i varied between alpha_i_min and alpha_i_max
    alpha_i_axis=ba.FixedBinAxis("alpha_i", 1, alpha_i*deg, alpha_i*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i_axis, 0.0*deg)
    simulation.setBeamIntensity(1e6)
    return simulation

def get_simulation_spec():
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=ba.SpecularSimulation()
    simulation.setBeamParameters(4.*pi*A, 1000, 0., 0.3)
    return simulation


# ## Complex simulation runs for GISANS with plotting

# In[4]:

def sim_and_show():
    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(up, 1.0, 0.5)
    simulation.runSimulation()
    uu=simulation.getIntensityData().getArray()

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.setBeamPolarization(down)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    simulation.runSimulation()
    dd=simulation.getIntensityData().getArray()

    sample=get_sample()
    simulation=get_simulation()
    simulation.setSample(sample)
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    simulation.runSimulation()
    ud=simulation.getIntensityData().getArray()

    Imax=max(uu.max(), dd.max(), ud.max())
    Imin=1e-4*Imax

    fig=pylab.figure(figsize=(18, 5))
    fig.suptitle(r'Simulations for $\lambda=%.2f~\AA$'%lambda_i, fontsize=18)
    pylab.subplot(131)
    # showing the result
    im=pylab.imshow(uu, norm=LogNorm(Imin, Imax),
                 extent=[2*pi/lambda_i*phi_f_min*deg, 2*pi/lambda_i*phi_f_max*deg,
                         2*pi/lambda_i*tth_min*deg, 2*pi/lambda_i*tth_max*deg],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.xlim(-0.04, 0.04)
    pylab.ylim(0., 0.1)
    pylab.xticks(linspace(-0.04, 0.04, 5))
    pylab.title(r'up-up', fontsize=14)
    cb.set_label(r'Intensity (arb. u.)', fontsize=14)
    pylab.xlabel(r'$Q_y~(\AA^{-1})$', fontsize=14)
    pylab.ylabel(r'$Q_z~(\AA^{-1})$', fontsize=14)

    # showing the result
    pylab.subplot(132)
    im=pylab.imshow(dd, norm=LogNorm(Imin, Imax),
                 extent=[2*pi/lambda_i*phi_f_min*deg, 2*pi/lambda_i*phi_f_max*deg,
                         2*pi/lambda_i*tth_min*deg, 2*pi/lambda_i*tth_max*deg],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.xlim(-0.04, 0.04)
    pylab.ylim(0., 0.1)
    pylab.xticks(linspace(-0.04, 0.04, 5))
    pylab.title(r'down-down', fontsize=14)
    cb.set_label(r'Intensity (arb. u.)', fontsize=14)
    pylab.xlabel(r'$Q_y~(\AA^{-1})$', fontsize=14)
    pylab.ylabel(r'$Q_z~(\AA^{-1})$', fontsize=14)

    # showing the result
    pylab.subplot(133)
    im=pylab.imshow(ud, norm=LogNorm(Imin, Imax),
                 extent=[2*pi/lambda_i*phi_f_min*deg, 2*pi/lambda_i*phi_f_max*deg,
                         2*pi/lambda_i*tth_min*deg, 2*pi/lambda_i*tth_max*deg],
                    aspect='auto', cmap='gist_ncar')
    cb=pylab.colorbar(im)
    pylab.xlim(-0.04, 0.04)
    pylab.ylim(0., 0.1)
    pylab.xticks(linspace(-0.04, 0.04, 5))
    pylab.title(r'up-down', fontsize=14)
    cb.set_label(r'Intensity (arb. u.)', fontsize=14)
    pylab.xlabel(r'$Q_y~(\AA^{-1})$', fontsize=14)
    pylab.ylabel(r'$Q_z~(\AA^{-1})$', fontsize=14)

    pylab.tight_layout(rect=(0, 0, 1, 0.95))


# # Functions to be called for the simulations

# In[5]:

def run_simulation_powder(only_first=False):
    """
    Run simulation and plot results
    """
    global lattice_rotation, lambda_i, up, down
    up=ba.kvector_t(0., 0., 1.)
    down=-up

    lambda_i=4.7
    sim_and_show()

    if only_first:
        return

    lambda_i=5.4
    sim_and_show()

    lambda_i=6.1
    sim_and_show()

    lambda_i=6.8
    sim_and_show()


def run_offspec_powder():
    """
    Run simulation and plot results
    """
    global lattice_rotation, alpha_i, lambda_i
    lambda_i=5.8
    alpha_i=0.3

    result=[]
    ais=[]
    pis=[]
    pfs=[]
    print u'Simulating αi=%.2f'%alpha_i
    for lambda_i in linspace(7.8, 2.6, 60):
        ais.append(alpha_i*ones(296/2))
        pis.append(2.*pi/lambda_i*alpha_i*deg*ones(296/2))
        pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296/2)*deg)
        sample=get_sample()
        simulation=get_simulation_offspec()
        simulation.setSample(sample)
        simulation.runSimulation()
        result.append(simulation.getIntensityData().getArray()[::-1, 0])

    for alpha_i in [0.54, 0.97, 1.75]:
        print u'Simulating αi=%.2f'%alpha_i
        for lambda_i in linspace(5.4, 2.6, 20):
            ais.append(alpha_i*ones(296/2))
            pis.append(2.*pi/lambda_i*alpha_i*deg*ones(296/2))
            pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296/2)*deg)
            sample=get_sample()
            simulation=get_simulation_offspec()
            simulation.setSample(sample)
            simulation.runSimulation()
            result.append(simulation.getIntensityData().getArray()[::-1, 0])

    result=array(result)*array(ais)/0.3
    pis=array(pis); pfs=array(pfs)

    Imax=result.max()
    Imin=Imax*1e-6

    pylab.figure(figsize=(4, 5))
    im=pylab.pcolormesh(pis-pfs, pis+pfs, result, norm=LogNorm(Imin, Imax))
    cb=pylab.colorbar(im)
    pylab.xlim(-0.05, 0.05)
    pylab.ylim(0., 0.15)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.xlabel(r'$p_i-p_f (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    #simulation=get_simulation_spec()
    #simulation.setSample(sample)
    #simulation.runSimulation()
    #result=simulation.getScalarR(0)
    #pylab.figure()
    #pylab.semilogy(abs(result))


if __name__=='__main__':
#  Ms=0.
#  alpha_i=0.35
#  run_simulation_powder(only_first=True)


  #Ms=0.
  #alpha_i=1.40
  #run_simulation_powder(only_first=True)


  #Ms=0.
  #run_offspec_powder()



  Ms=2.e0
  alpha_i=0.35
  run_simulation_powder(only_first=True)

  pylab.show()