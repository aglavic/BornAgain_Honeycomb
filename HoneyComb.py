#-*- coding: utf-8 -*-
"""
Main script to run the simulation for the Honeycomb lattice sample.

Author: Artur Glavic

BornAgain verion 1.7.1
"""
from numpy import sin, cos, arange, pi, linspace, array, ones, sqrt, meshgrid, loadtxt, exp, savetxt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import pylab
import bornagain as ba

from matplotlib import rcParams
rcParams['mathtext.default']='regular'
rcParams["font.size"]="12"

from HoneyCombSample import HCSample

A=ba.angstrom
nm=ba.nm
deg=ba.degree
Ms=0.
MsB=15.
lattice_rotation=0.
is_FM=True

INTEGRATE_XI=False

alpha_i_min, alpha_i_max=0.0, 4.0
alpha_i=0.35
lambda_i=5.5

phi_f_min, phi_f_max=-1.9, 1.75
alpha_f_min, alpha_f_max=0.0, 4.0
tth_min, tth_max=-0.08, 4.4
phi_f_offspec=0.3

s3=sqrt(3.)

BIN=2



def get_simulation():
    """
    Create and return GISAXS simulation with beam and detector defined
    """
    simulation=ba.GISASSimulation()
    simulation.setDetectorParameters(248/BIN, phi_f_min*deg, phi_f_max*deg, 296/BIN,
                                     (tth_min-alpha_i)*deg, (tth_max-alpha_i)*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i*deg, 0.0*deg)
    simulation.setBeamIntensity(1e6)

    if INTEGRATE_XI:
      xi_rotation=ba.DistributionGate(-90, 90)
      simulation.addParameterDistribution("*/SampleBuilder/xi", xi_rotation, 36)

    return simulation

def get_simulation_offspec():
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=ba.OffSpecSimulation()
    simulation.setDetectorParameters(11,-phi_f_offspec*deg, phi_f_offspec,
                                     296/BIN, alpha_f_min*deg, alpha_f_max*deg)
    # defining the beam  with incidence alpha_i varied between alpha_i_min and alpha_i_max
    alpha_i_axis=ba.FixedBinAxis("alpha_i", 1, alpha_i*deg, alpha_i*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i_axis, 0.0*deg)
    simulation.setBeamIntensity(1e6)

    if INTEGRATE_XI:
      xi_rotation=ba.DistributionGate(-90, 90)
      simulation.addParameterDistribution("*/SampleBuilder/xi", xi_rotation, 36)

    return simulation

def get_simulation_offspec_angular(ai_min, ai_max, nbins):
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=ba.OffSpecSimulation()
    simulation.setDetectorParameters(11,-phi_f_offspec*deg, phi_f_offspec,
                                     296/BIN, alpha_f_min*deg, alpha_f_max*deg)
    # defining the beam  with incidence alpha_i varied between alpha_i_min and alpha_i_max
    alpha_i_axis=ba.FixedBinAxis("alpha_i", nbins, ai_min*deg, ai_max*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i_axis, 0.0*deg)
    simulation.setBeamIntensity(1e6)

    if INTEGRATE_XI:
      xi_rotation=ba.DistributionGate(-90, 90)
      simulation.addParameterDistribution("*/SampleBuilder/xi", xi_rotation, 36)

    return simulation

def get_simulation_spec():
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=ba.SpecularSimulation()
    simulation.setBeamParameters(4.*pi*A, 1000, 0., 0.3)
    return simulation


def sim_and_show(model='ferro'):
    SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                magnetic_model=model, xi=lattice_rotation)
    simulation=get_simulation()
    simulation.setSampleBuilder(SB)
    #simulation.printParameters()
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(up, 1.0, 0.5)
    simulation.runSimulation()
    uu=simulation.getIntensityData().getArray()

    SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                magnetic_model=model, xi=lattice_rotation)
    simulation=get_simulation()
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(down)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    simulation.runSimulation()
    dd=simulation.getIntensityData().getArray()

    SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                magnetic_model=model, xi=lattice_rotation)
    simulation=get_simulation()
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    simulation.runSimulation()
    ud=simulation.getIntensityData().getArray()

    Imax=1e2#max(uu.max(), dd.max(), ud.max())
    Imin=1e-4*Imax

    fig=pylab.figure(figsize=(18, 5))
    fig.suptitle(r'Simulations for $\lambda=%.2f~\AA$'%lambda_i, fontsize=18)
    pylab.subplot(131)
    # showing the result
    im=pylab.imshow((uu+dd)/2., norm=LogNorm(Imin, Imax),
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
    im=pylab.imshow(uu-dd, norm=LogNorm(Imin, Imax),
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


def run_simulation_powder(only_first=False, model='ferro'):
    """
    Run simulation and plot results
    """
    global lattice_rotation, lambda_i, up, down
    up=ba.kvector_t(0., 1., 0.)
    down=-up

    lambda_i=4.7
    sim_and_show(model=model)

    if only_first:
        return

    lambda_i=5.4
    sim_and_show()

    lambda_i=6.1
    sim_and_show()

    lambda_i=6.8
    sim_and_show()


def run_offspec_powder(model='ferro'):
    """
    Run simulation and plot results
    """
    global lattice_rotation, alpha_i, lambda_i
    lambda_i=5.8
    alpha_i=0.3
    up=ba.kvector_t(0., 1., 0.)
    down=-up

    result=[]
    ais=[]
    pis=[]
    pfs=[]
    lis=[]
    print u'Simulating αi=%.2f'%alpha_i
    for lambda_i in linspace(7.8, 2.6, 60):
      lis.append(lambda_i*ones(296/BIN))
      ais.append(alpha_i*ones(296/BIN))
      pis.append(2.*pi/lambda_i*alpha_i*deg*ones(296/BIN))
      pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296/BIN)*deg)
      SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i, magnetic_model=model)
      simulation=get_simulation_offspec()
      simulation.setSampleBuilder(SB)
      simulation.setBeamPolarization(up)
      simulation.setAnalyzerProperties(up, 1.0, 0.5)
      simulation.runSimulation()
      result.append(simulation.getIntensityData().getArray()[::-1, 0])

    for alpha_i in [0.54, 0.97, 1.75]:
      print u'Simulating αi=%.2f'%alpha_i
      lambda_i=4.0
      for lambda_i in linspace(5.4, 2.6, 20):
        lis.append(lambda_i*ones(296/BIN))
        ais.append(alpha_i*ones(296/BIN))
        pis.append(2.*pi/lambda_i*alpha_i*deg*ones(296/BIN))
        pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296/BIN)*deg)
        SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i, magnetic_model=model)
        simulation=get_simulation_offspec()
        simulation.setSampleBuilder(SB)
        simulation.setBeamPolarization(up)
        simulation.setAnalyzerProperties(up, 1.0, 0.5)
        simulation.runSimulation()
        result.append(simulation.getIntensityData().getArray()[::-1, 0])

    result=array(result)*(array(ais)/0.3)#**2/array(lis)##
    pis=array(pis); pfs=array(pfs)

    Imax=result.max()
    Imin=Imax*1e-8

    pylab.figure(figsize=(5.5, 7))
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

def run_offspec_powder_angular(model='ferro', save=None):
    """
    Run simulation and plot results
    """
    global lattice_rotation, alpha_i, lambda_i
    up=ba.kvector_t(0., 1., 0.)
    down=-up

    lambda_i=4.0
    alpha_i=0.3

    qu, ru=loadtxt('data/fit_upd_model_4_000.dat').T[0:2]
    qd, rd=loadtxt('data/fit_upd_model_4_001.dat').T[0:2]
    ru_q=interp1d(qu, ru, bounds_error=False, fill_value=1.0)
    rd_q=interp1d(qd, rd, bounds_error=False, fill_value=1.0)

    SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                magnetic_model=model)
    simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN)
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(up)
    simulation.setAnalyzerProperties(up, 1.0, 0.5)
    simulation.runSimulation()
    uu=simulation.getIntensityData().getArray()[::-1]

    SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                magnetic_model=model)
    simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN)
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(down)
    simulation.setAnalyzerProperties(down, 1.0, 0.5)
    simulation.runSimulation()
    dd=simulation.getIntensityData().getArray()[::-1]
    
    if model!='ferro':
      SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                  magnetic_model=model)
      simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN)
      simulation.setSampleBuilder(SB)
      simulation.setBeamPolarization(up)
      simulation.setAnalyzerProperties(down, 1.0, 0.5)
      simulation.runSimulation()
      ud=simulation.getIntensityData().getArray()[::-1]

      SB=HCSample(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                  magnetic_model=model)
      simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN)
      simulation.setSampleBuilder(SB)
      simulation.setBeamPolarization(down)
      simulation.setAnalyzerProperties(up, 1.0, 0.5)
      simulation.runSimulation()
      du=simulation.getIntensityData().getArray()[::-1]

      fig=pylab.figure(figsize=(13.5, 6))
      fig.suptitle(r'Simulations for vortex state', fontsize=18)
      spn=130
    else:
      fig=pylab.figure(figsize=(9, 6))
      fig.suptitle(r'Simulations for ferromagnetic state', fontsize=18)
      spn=120

    k=2.*pi/lambda_i
    ais=linspace(0.05, 2.5, 200/BIN)
    afs=linspace(alpha_f_min, alpha_f_max, 296/BIN)
    pis, pfs=meshgrid(k*ais*deg, k*afs*deg)
    uu*=ais/0.05/BIN
    dd*=ais/0.05/BIN
    Ru=ru_q(pis+pfs)*exp(-0.5*((pis-pfs)/0.0004)**2)*2e6
    Rd=rd_q(pis+pfs)*exp(-0.5*((pis-pfs)/0.0004)**2)*2e6
    Intensities=[Ru, Rd, uu, dd]
    header='# pi-pf Qz Ru Rd uu dd'
    if model!='ferro':
      ud*=ais/0.05/BIN
      du*=ais/0.05/BIN
      Intensities.append(ud)
      Intensities.append(du)
      header+=' ud du'

    Imax=uu.max()
    Imin=Imax*1e-6

    pylab.subplot(spn+1)
    pylab.title('NSF [(++)+(--)]')
    im=pylab.pcolormesh(pis-pfs, pis+pfs, (uu+dd)/2., norm=LogNorm(Imin, Imax), cmap='gist_ncar')
    pylab.xlim(-0.05, 0.05)
    pylab.ylim(0., 0.15)
    pylab.xlabel(r'$p_i-p_f (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    pylab.subplot(spn+2)
    pylab.title('FM [(++)-(--)]')
    im=pylab.pcolormesh(pis-pfs, pis+pfs, uu-dd, norm=LogNorm(Imin, Imax), cmap='gist_ncar')
    pylab.xlim(-0.05, 0.05)
    pylab.ylim(0., 0.15)
    pylab.xlabel(r'$p_i-p_f (\AA^{-1})$', fontsize=16)
    pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    if model!='ferro':
      pylab.subplot(spn+3)
      pylab.title('SF [(+-)+(-+)]')
      im=pylab.pcolormesh(pis-pfs, pis+pfs, (ud+du)/2., norm=LogNorm(Imin, Imax), cmap='gist_ncar')
      pylab.xlim(-0.05, 0.05)
      pylab.ylim(0., 0.15)
      pylab.xlabel(r'$p_i-p_f (\AA^{-1})$', fontsize=16)
      pylab.ylabel(r'$Q_z (\AA^{-1})$', fontsize=16)

    cb=pylab.colorbar(im)
    cb.set_label(r'Intensity (arb. u.)', fontsize=16)
    pylab.tight_layout(rect=(0, 0, 1, 0.95))

    if save is not None:
      fh=open(save, 'w')
      fh.write(header+'\n')
      for items in zip(*tuple([pis-pfs, pis+pfs]+Intensities)):
        savetxt(fh, array(list(items)).T)
        fh.write('\n')




if __name__=='__main__':
#  Ms=0.
#  alpha_i=0.35
#  run_simulation_powder(only_first=True)


  #Ms=0.
  #alpha_i=1.40
  #run_simulation_powder(only_first=True)


  INTEGRATE_XI=True
  Ms=0.
  MsB=30.
  run_offspec_powder_angular(model='vortex', save='data/offspec_simulation.dat')
  Ms=20.
  MsB=5.
  run_offspec_powder_angular(model='vortex', save='data/offspec_simulation_vortex.dat')



  #Ms=0.
  #MsB=15.
  #alpha_i=0.35
  #run_simulation_powder(only_first=True)
  #run_simulation_powder(only_first=True, model='vortex')

  #Ms=15.
  #MsB=5.
  #alpha_i=0.35
  #run_simulation_powder(only_first=True, model='vortex')

  #pylab.show()
