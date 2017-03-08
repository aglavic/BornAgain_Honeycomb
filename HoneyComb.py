#-*- coding: utf-8 -*-
"""
Main script to run the simulation for the Honeycomb lattice sample.

Author: Artur Glavic

BornAgain version 1.7.1
"""
from numpy import pi, linspace, array, ones, meshgrid, loadtxt, exp, savetxt
from scipy.interpolate import interp1d
import pylab
import bornagain as ba

from HoneyCombSample import HCSample
from HoneyCombPlotting import BAPlotter

A=ba.angstrom
nm=ba.nm
deg=ba.degree

Ms=0.
MsB=15.
lattice_rotation=0.
INTEGRATE_XI=False

alpha_i_min, alpha_i_max=0.0, 4.0
alpha_i=0.35
lambda_i=5.5

phi_f_min, phi_f_max=-1.9, 1.75
alpha_f_min, alpha_f_max=0.0, 4.0
tth_min, tth_max=-0.08, 4.4
phix_f_min, phix_f_max=-0.8, 2.2
tthx_min, tthx_max=0.0, 2.0
phi_f_offspec=0.3

BIN=4 # reduce resolution by this factor to speed up simulation

# neutron spin states
_up=ba.kvector_t(0., 1., 0.)
_down=-_up

CHANNELS={'uu': (_up, _up),
          'dd': (_down, _down),
          'ud': (_up, _down),
          'du': (_down, _up)}

def get_simulation_gisans(channel, **kwrds):
    """
    Create and return GISANS simulation with beam and detector defined
    """
    simulation=ba.GISASSimulation()
    simulation.setDetectorParameters(248/BIN, phi_f_min*deg, phi_f_max*deg, 296/BIN,
                                     (tth_min-alpha_i)*deg, (tth_max-alpha_i)*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i*deg, 0.0*deg)
    simulation.setBeamIntensity(1e6)

    if INTEGRATE_XI:
      xi_rotation=ba.DistributionGate(-90, 90)
      simulation.addParameterDistribution("*/SampleBuilder/xi", xi_rotation, 36)

    SB=HCSample(**kwrds)
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(CHANNELS[channel][0])
    simulation.setAnalyzerProperties(CHANNELS[channel][1], 1.0, 0.5)

    return simulation

def get_simulation_gisaxs(**kwrds):
    """
    Create and return GISAXS simulation with beam and detector defined
    """
    simulation=ba.GISASSimulation()
    simulation.setDetectorParameters(512/BIN, phix_f_min*deg, phix_f_max*deg, 512/BIN,
                                    (tthx_min-alpha_i)*deg, (tthx_max-alpha_i)*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i*deg, 0.0*deg)
    simulation.setBeamIntensity(1e6)

    if INTEGRATE_XI:
      xi_rotation=ba.DistributionGate(-90, 90)
      simulation.addParameterDistribution("*/SampleBuilder/xi", xi_rotation, 36)

    SB=HCSample(**kwrds)
    simulation.setSampleBuilder(SB)

    return simulation

def get_simulation_offspec(channel, **kwrds):
    """
    Create and return off-specular simulation with beam and detector defined
    """
    simulation=ba.OffSpecSimulation()
    simulation.setDetectorParameters(11,-phi_f_offspec*deg, phi_f_offspec,
                                     296/BIN, alpha_f_min*deg, alpha_f_max*deg)

    alpha_i_axis=ba.FixedBinAxis("alpha_i", 1, alpha_i*deg, alpha_i*deg)
    simulation.setBeamParameters(lambda_i*A, alpha_i_axis, 0.0*deg)
    simulation.setBeamIntensity(1e6)

    if INTEGRATE_XI:
      xi_rotation=ba.DistributionGate(-90, 90)
      simulation.addParameterDistribution("*/SampleBuilder/xi", xi_rotation, 36)

    SB=HCSample(**kwrds)
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(CHANNELS[channel][0])
    simulation.setAnalyzerProperties(CHANNELS[channel][1], 1.0, 0.5)

    return simulation

def get_simulation_offspec_angular(ai_min, ai_max, nbins, channel='uu', **kwrds):
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

    SB=HCSample(**kwrds)
    simulation.setSampleBuilder(SB)
    simulation.setBeamPolarization(CHANNELS[channel][0])
    simulation.setAnalyzerProperties(CHANNELS[channel][1], 1.0, 0.5)

    return simulation

def get_simulation_spec(channel, **kwrds):
    """
    Create and return specular simulation with beam and detector defined.
    
    Not yet working.
    """
    simulation=ba.SpecularSimulation()
    simulation.setBeamParameters(lambda_i*A, 1000, alpha_i_min*deg, alpha_i_max*deg)

    SB=HCSample(**kwrds)
    simulation.setSampleBuilder(SB)

    # this does not work!!!
    simulation.setBeamPolarization(CHANNELS[channel][0])
    simulation.setAnalyzerProperties(CHANNELS[channel][1], 1.0, 0.5)

    return simulation


def sim_and_show_gisans(model='ferro', save=None):
    """
    Simulate all GISANS channels and plot the result.
    """
    print "GISANS %s at %.2fdeg and %.2fA"%(model, alpha_i, lambda_i)

    kwrds=dict(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                basic_model=model, xi=lattice_rotation)

    simulation=get_simulation_gisans('uu', **kwrds)
    simulation.runSimulation()
    uu=simulation.getIntensityData().getArray()

    simulation=get_simulation_gisans('dd', **kwrds)
    simulation.runSimulation()
    dd=simulation.getIntensityData().getArray()

    simulation=get_simulation_gisans('ud', **kwrds)
    simulation.runSimulation()
    ud=simulation.getIntensityData().getArray()

    simulation=get_simulation_gisans('du', **kwrds)
    simulation.runSimulation()
    du=simulation.getIntensityData().getArray()

    Imax=max(uu.max(), dd.max(), ud.max(), du.max())
    Imin=1e-5*Imax

    plt=BAPlotter(nitems=3, rows=1, title=r'GISANS for $\lambda$=%.2f $\AA$'%lambda_i,
                  gisans=True, xlim=(-0.05, 0.05), ylim=(0., 0.1))
    extent=[2*pi/lambda_i*phi_f_min*deg, 2*pi/lambda_i*phi_f_max*deg,
            2*pi/lambda_i*tth_min*deg, 2*pi/lambda_i*tth_max*deg]
    # showing the result
    plt.imshow((uu+dd)/2., extent, Imin=Imin, Imax=Imax, title='NSF [(++)+(--)]')
    plt.imshow(uu-dd, extent, Imin=Imin, Imax=Imax, title='FM [(++)-(--)]')
    plt.imshow((ud+du)/2., extent, Imin=Imin, Imax=Imax, title='SF [(+-)+(-+)]')

    if save is not None:
      fn=save.rsplit('.', 1)[0]+('_%.2fdeg_%.2fA.'%(lambda_i, alpha_i))+save.rsplit('.', 1)[1]
      fh=open(fn, 'w')
      header='# Qy Qz Iuu Idd Iud Idu'
      fh.write(header+'\n')
      Qy, Qz=meshgrid(linspace(2*pi/lambda_i*phi_f_min*deg, 2*pi/lambda_i*phi_f_max*deg, 248/BIN),
                     linspace(2*pi/lambda_i*tth_min*deg, 2*pi/lambda_i*tth_max*deg, 296/BIN))
      for items in zip(*tuple([Qy, Qz, uu[::-1], dd[::-1], ud[::-1], du[::-1]])):
        savetxt(fh, array(list(items)).T)
        fh.write('\n')


def run_gisans(only_first=False, model='ferro', save=None):
    """
    Run simulation for GISANS and plot result
    """
    global lattice_rotation, lambda_i, up, down, alpha_i
    up=ba.kvector_t(0., 1., 0.)
    down=-up

    alpha_i=0.35
    lambda_i=4.7
    sim_and_show_gisans(model=model, save=save)

    if only_first:
        return

    lambda_i=5.4
    sim_and_show_gisans(model=model, save=save)

    lambda_i=6.1
    sim_and_show_gisans(model=model, save=save)

    lambda_i=6.8
    sim_and_show_gisans(model=model, save=save)

    alpha_i=1.4
    lambda_i=4.7
    sim_and_show_gisans(model=model, save=save)

    lambda_i=5.4
    sim_and_show_gisans(model=model, save=save)

    lambda_i=6.1
    sim_and_show_gisans(model=model, save=save)

    lambda_i=6.8
    sim_and_show_gisans(model=model, save=save)

def run_gisaxs(ai=0.25, save=None):
    """
    Run simulation for GISANS and plot result
    """
    global lattice_rotation, lambda_i, alpha_i
    alpha_i=ai
    lambda_i=1.54

    kwrds=dict(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
              hc_lattice_length=27.7*nm, hc_inner_radius=10.0*nm,
              domain=300.*nm, cauchy=1.*nm,
              py_d=16.0*nm, top_d=0.0*nm,
              basic_model='xray', xi=lattice_rotation)

    simulation=get_simulation_gisaxs(**kwrds)
    simulation.runSimulation()
    I=simulation.getIntensityData().getArray()

    Imax=I.max()
    Imin=1e-7*Imax

    plt=BAPlotter(nitems=1, rows=1, title=r'GISAXS for $\lambda$=%.2f $\AA$'%lambda_i,
                  gisans=True, xlim=(-0.05, 0.15), ylim=(0, 0.1))
    extent=[2*pi/lambda_i*phix_f_min*deg, 2*pi/lambda_i*phix_f_max*deg,
            2*pi/lambda_i*tthx_min*deg, 2*pi/lambda_i*tthx_max*deg]
    # showing the result
    plt.imshow(I, extent, Imin=Imin, Imax=Imax, title='')

    if save is not None:
      fh=open(save, 'w')
      header='# Qy Qz Ix'
      fh.write(header+'\n')
      Qy, Qz=meshgrid(linspace(2*pi/lambda_i*phix_f_min*deg, 2*pi/lambda_i*phix_f_max*deg, 512/BIN),
                     linspace(2*pi/lambda_i*tthx_min*deg, 2*pi/lambda_i*tthx_max*deg, 512/BIN))
      for items in zip(*tuple([Qy, Qz, I[::-1]])):
        savetxt(fh, array(list(items)).T)
        fh.write('\n')

def run_offspec(model='ferro', save=None):
    """
    Run simulation for ToF off-specular and plot results
    """
    global lattice_rotation, alpha_i, lambda_i
    lambda_i=5.8
    alpha_i=0.3

    qu, ru=loadtxt('data/fit_upd_model_4_000.dat').T[0:2]
    qd, rd=loadtxt('data/fit_upd_model_4_001.dat').T[0:2]
    ru_q=interp1d(qu, ru, bounds_error=False, fill_value=1.0)
    rd_q=interp1d(qd, rd, bounds_error=False, fill_value=1.0)
    Ru=[]
    Rd=[]
    kwrds=dict(bias_field=MsB, edge_field=Ms, basic_model=model)

    uu=[]; dd=[]; ud=[]; du=[]
    ais=[]; pis=[]; pfs=[]; lis=[]
    print('Simulating αi=%.2f'%alpha_i)
    for lambda_i in linspace(7.8, 2.6, 60):
      kwrds['lambda_i']=lambda_i
      lis.append(lambda_i*ones(296/BIN))
      ais.append(alpha_i*ones(296/BIN))
      pis.append(2.*pi/lambda_i*alpha_i*deg*ones(296/BIN))
      pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296/BIN)*deg)

      simulation=get_simulation_offspec('uu', **kwrds)
      simulation.runSimulation()
      uu.append(simulation.getIntensityData().getArray()[::-1, 0])
      simulation=get_simulation_offspec('dd', **kwrds)
      simulation.runSimulation()
      dd.append(simulation.getIntensityData().getArray()[::-1, 0])
      if model!='ferro':
        simulation=get_simulation_offspec('ud', **kwrds)
        simulation.runSimulation()
        ud.append(simulation.getIntensityData().getArray()[::-1, 0])
        simulation=get_simulation_offspec('du', **kwrds)
        simulation.runSimulation()
        du.append(simulation.getIntensityData().getArray()[::-1, 0])

    for alpha_i in [0.54, 0.97, 1.75]:
      print('Simulating αi=%.2f'%alpha_i)
      for lambda_i in linspace(5.4, 2.6, 20):
        kwrds['lambda_i']=lambda_i
        lis.append(lambda_i*ones(296/BIN))
        ais.append(alpha_i*ones(296/BIN))
        pis.append(2.*pi/lambda_i*alpha_i*deg*ones(296/BIN))
        pfs.append(2.*pi/lambda_i*linspace(alpha_f_min, alpha_f_max, 296/BIN)*deg)

        simulation=get_simulation_offspec('uu', **kwrds)
        simulation.runSimulation()
        uu.append(simulation.getIntensityData().getArray()[::-1, 0])
        simulation=get_simulation_offspec('dd', **kwrds)
        simulation.runSimulation()
        dd.append(simulation.getIntensityData().getArray()[::-1, 0])
        if model!='ferro':
          simulation=get_simulation_offspec('ud', **kwrds)
          simulation.runSimulation()
          ud.append(simulation.getIntensityData().getArray()[::-1, 0])
          simulation=get_simulation_offspec('du', **kwrds)
          simulation.runSimulation()
          du.append(simulation.getIntensityData().getArray()[::-1, 0])

    uu=array(uu)*(array(ais)/0.3)#*array(lis)**2## Why do I need to scale by λ²?
    dd=array(dd)*(array(ais)/0.3)#*array(lis)**2##
    pis=array(pis); pfs=array(pfs)
    Ru=ru_q(pis+pfs)*exp(-0.5*((pis-pfs)/0.0004)**2)*2e6
    Rd=rd_q(pis+pfs)*exp(-0.5*((pis-pfs)/0.0004)**2)*2e6
    header='# pi-pf Qz Ru Rd uu dd'
    Intensities=[Ru, Rd, uu, dd]
    if model!='ferro':
      ud=array(ud)*(array(ais)/0.3)#*array(lis)**2## Why do I need to scale by λ²?
      du=array(ud)*(array(ais)/0.3)#*array(lis)**2##
      header+=' ud du'
      Intensities+=[ud, du]
      plt=BAPlotter(nitems=3, title=r'ToF simulations for vortex state',
                    gisans=False, xlim=(-0.05, 0.05), ylim=(0., 0.15))
    else:
      plt=BAPlotter(nitems=2, title=r'ToF simulations for ferromagnetic state',
                    gisans=False, xlim=(-0.05, 0.05), ylim=(0., 0.15))

    Imax=max(uu.max(), dd.max())
    Imin=Imax*1e-7


    plt.pcolormesh(pis-pfs, pis+pfs, (uu+dd)/2., title='NSF [(++)+(--)]', Imin=Imin, Imax=Imax)
    plt.pcolormesh(pis-pfs, pis+pfs, uu-dd, title='FM [(++)-(--)]', Imin=Imin, Imax=Imax)

    if model!='ferro':
      plt.pcolormesh(pis-pfs, pis+pfs, (ud+du)/2., title='SF [(+-)+(-+)]', Imin=Imin, Imax=Imax)

    if save is not None:
      fh=open(save, 'w')
      fh.write(header+'\n')
      for items in zip(*tuple([pis-pfs, pis+pfs]+Intensities)):
        savetxt(fh, array(list(items)).T)
        fh.write('\n')

def run_offspec_angular(model='ferro', save=None):
    """
    Run simulation for angular dispersive off-specular and plot results
    """
    global lattice_rotation, alpha_i, lambda_i
    print "OffSpecular (angular) %s"%(model)


    lambda_i=4.0
    alpha_i=0.3

    qu, ru=loadtxt('data/fit_upd_model_4_000.dat').T[0:2]
    qd, rd=loadtxt('data/fit_upd_model_4_001.dat').T[0:2]
    ru_q=interp1d(qu, ru, bounds_error=False, fill_value=1.0)
    rd_q=interp1d(qd, rd, bounds_error=False, fill_value=1.0)

    kwrds=dict(bias_field=MsB, edge_field=Ms, lambda_i=lambda_i,
                basic_model=model)


    simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN, 'uu', **kwrds)
    simulation.runSimulation()
    uu=simulation.getIntensityData().getArray()[::-1]
    simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN, 'dd', **kwrds)
    simulation.runSimulation()
    dd=simulation.getIntensityData().getArray()[::-1]
    
    if model!='ferro':
      simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN, 'ud', **kwrds)
      simulation.runSimulation()
      ud=simulation.getIntensityData().getArray()[::-1]
      simulation=get_simulation_offspec_angular(0.05, 2.5, 200/BIN, 'du', **kwrds)
      simulation.runSimulation()
      du=simulation.getIntensityData().getArray()[::-1]

      plt=BAPlotter(nitems=3, title=r'Simulations for vortex state',
                    gisans=False, xlim=(-0.05, 0.05), ylim=(0., 0.15))
    else:
      plt=BAPlotter(nitems=2, title=r'Simulations for ferromagnetic state',
                    gisans=False, xlim=(-0.05, 0.05), ylim=(0., 0.15))

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
    Imin=Imax*1e-7

    plt.pcolormesh(pis-pfs, pis+pfs, (uu+dd)/2., title='NSF [(++)+(--)]', Imin=Imin, Imax=Imax)
    plt.pcolormesh(pis-pfs, pis+pfs, uu-dd, title='FM [(++)-(--)]', Imin=Imin, Imax=Imax)

    if model!='ferro':
      plt.pcolormesh(pis-pfs, pis+pfs, (ud+du)/2., title='SF [(+-)+(-+)]', Imin=Imin, Imax=Imax)

    if save is not None:
      fh=open(save, 'w')
      fh.write(header+'\n')
      for items in zip(*tuple([pis-pfs, pis+pfs]+Intensities)):
        savetxt(fh, array(list(items)).T)
        fh.write('\n')

def debug(model='ferro'):
  s=HCSample(basic_model=model)
  s.buildSample()


if __name__=='__main__':
  #debug(model='ice-2') # call that throws actual usable python error

  INTEGRATE_XI=True # should be used for actual simulation, False will speed it up very much
  Ms=0.
  MsB=0.10e6
  run_offspec_angular(model='ferro', save='data/offspec_ferro.dat')
  run_offspec(model='ferro', save='data/offspec_tof_ferro.dat')
#  Ms=1.5e6
#  MsB=1e3
#  run_offspec_angular(model='vortex', save='data/offspec_vortex.dat') # should be run for actual simulation to get rid of differences due to unit cell size
#
#  INTEGRATE_XI=True # should be used for actual simulation, False will speed it up very much
#  Ms=0.
#  MsB=0.03e6
#  print "Simulating ferro model"
#  run_gisans(only_first=True, model='ferro', save='data/GISANS_ferro.dat')
#  Ms=1.5e6
#  MsB=0.001e6
#  run_gisans(only_first=False, model='glass', save='data/GISANS_glass.dat')
#  run_gisans(only_first=True, model='ice-1', save='data/GISANS_ice-1.dat')
#  run_gisans(only_first=True, model='ice-2', save='data/GISANS_ice-2.dat')
#  run_gisans(only_first=False, model='vortex', save='data/GISANS_vortex.dat')
  #run_gisaxs()

  # this does not work!!!
  #simulation=get_simulation_spec('uu')
  #simulation.runSimulation()
  #pylab.semilogy(simulation.getScalarR(0))

  pylab.show()
