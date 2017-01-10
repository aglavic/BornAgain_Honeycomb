#-*- coding: utf-8 -*-
'''
Help module to do the plotting of simulation results.

Author: Artur Glavic

BornAgain version 1.7.1
'''

from matplotlib.colors import LogNorm, Normalize
import pylab

from matplotlib import rcParams
rcParams['mathtext.default']='regular'
rcParams["font.size"]="12"


class BAPlotter(object):

  xlabel_offspec=r'$Q_y$ ($\AA^{-1}$)'
  xlabel_gisans=r'$Q_x$ ($\AA^{-1}$)'
  ylabel=r'$Q_z$ ($\AA^{-1}$)'
  clabel=r'Intensity (arb. u.)'

  def __init__(self, nitems=2, rows=1, title='', gisans=True, xlim=None, ylim=None,
               cmap='gist_ncar'):
    self.nitems=nitems
    self.index=1
    self.subnum=100*rows+10*nitems

    self.xlim=xlim
    self.ylim=ylim
    self.cmap=cmap
    
    if gisans:
      self.fig=pylab.figure(figsize=(6*nitems, 5))
      self.xlabel=self.xlabel_gisans
    else:
      self.fig=pylab.figure(figsize=(4.5*nitems, 5))
      self.xlabel=self.xlabel_offspec

    self.fig.suptitle(title, fontsize=16)

  def subplot(self):
    if self.xlim is not None and self.ylim is not None:
      pylab.subplot(self.subnum+self.index, adjustable='box', aspect=1.0)
    else:
      pylab.subplot(self.subnum+self.index)
    self.index+=1
  
  def pcolormesh(self, x, y, I, log=True, Imin=None, Imax=None, title='', **opts):
    if Imin is None:
      Imin=I[I>0].min()
    if Imax is None:
      Imax=I.max()
    if log:
      norm=LogNorm(Imin, Imax)
    else:
      norm=Normalize(Imin, Imax)

    self.subplot()
    im=pylab.pcolormesh(x, y, I, norm=norm, cmap=self.cmap, **opts)

    if self.xlim is not None:
      pylab.xlim(*self.xlim)
    if self.ylim is not None:
      pylab.ylim(*self.ylim)

    pylab.title(title, fontsize=12)
    pylab.xlabel(self.xlabel, fontsize=12)
    pylab.ylabel(self.ylabel, fontsize=12)


    if self.index>self.nitems:
      cb=pylab.colorbar(im)
      cb.set_label(self.clabel, fontsize=12)
      pylab.tight_layout(rect=(0, 0, 1, 0.95))

  def imshow(self, I, extent, log=True, Imin=None, Imax=None, title='', **opts):
    if Imin is None:
      Imin=I[I>0].min()
    if Imax is None:
      Imax=I.max()
    if log:
      norm=LogNorm(Imin, Imax)
    else:
      norm=Normalize(Imin, Imax)

    self.subplot()
    im=pylab.imshow(I, norm=norm, aspect=1.0, cmap=self.cmap, extent=extent, **opts)

    if self.xlim is not None:
      pylab.xlim(*self.xlim)
    if self.ylim is not None:
      pylab.ylim(*self.ylim)

    pylab.title(title, fontsize=12)
    pylab.xlabel(self.xlabel, fontsize=12)
    pylab.ylabel(self.ylabel, fontsize=12)


    if self.index==self.nitems:
      cb=pylab.colorbar(im)
      cb.set_label(self.clabel, fontsize=12)
      pylab.tight_layout(rect=(0, 0, 1, 0.95))
  
