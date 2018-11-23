#!/usr/bin/env python
'''
% Copyright (c) 2011-2015 by Thomas O'Hara, Yoram Rudy,
%                            Washington University in St. Louis.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution.
%
% 3. Neither the names of the copyright holders nor the names of its
% contributors may be used to endorse or promote products derived from
% this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
% USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
% THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
% DAMAGE.

% MATLAB Implementation of the O'Hara-Rudy dynamic (ORd) model for the
% undiseased human ventricular action potential and calcium transient
%
% The ORd model is described in the article "Simulation of the Undiseased
% Human Cardiac Ventricular Action Potential: Model Formulation and
% Experimental Validation"
% by Thomas O'Hara, Laszlo Virag, Andras Varro, and Yoram Rudy
%
% The article and supplemental materails are freely available in the
% Open Access jounal PLoS Computational Biology
% Link to Article:
% http://www.ploscompbiol.org/article/info:doi/10.1371/journal.pcbi.1002061
%
% Email: tom.ohara@gmail.com / rudy@wustl.edu
% Web: http://rudylab.wustl.edu
%
% Implemented in Python by William A Coetzee on 11/20/2018
%
'''

__debugging__ = False

import time
import scipy as sp
import pylab as plt
import numpy as np
from numpy import array
from scipy.integrate import odeint
from ohara_model import ohara

if __debugging__:
    import pdb


def Main(self):
    """
    Main file for the O'Hara human ventricular model
        A method of class ohara in 'ohara_model.py'
        calls function dALLdt() for the ODE

    """

    # Keep all data and/or save to disk?
    SaveAll = True
    Savetodisk = True

    start_time = time.time()
    self.print()

    # Values for initial conditions
    v = -87.0
    nai = 7.0
    nass = nai
    ki = 145.0
    kss = ki
    cai = 1.0e-4
    cass = cai
    cansr = 1.2
    cajsr = cansr
    m = 0.0
    hf = 1.0
    hs = 1.0
    j = 1.0
    hsp = 1.0
    jp = 1.0
    mL = 0.0
    hL = 1.0
    hLp = 1.0
    a = 0.0
    iF = 1.0
    iS = 1.0
    ap = 0.0
    iFp = 1.0
    iSp = 1.0
    d = 0.0
    ff = 1.0
    fs = 1.0
    fcaf = 1.0
    fcas = 1.0
    jca = 1.0
    nca = 0.0
    ffp = 1.0
    fcafp = 1.0
    xrf = 0.0
    xrs = 0.0
    xs1 = 0.0
    xs2 = 0.0
    xk1 = 1.0
    Jrelnp = 0.0
    Jrelp = 0.0
    CaMKt = 0.0

    cell_type = {
        0: 'Endocardial',
        1: 'Epicardial',
        2: 'M-cell'}

    # X0 is the vector for initial sconditions for state variables
    X0 = [v, nai, nass, ki, kss, cai, cass, cansr, cajsr, m, hf, hs, j, hsp, jp, mL, hL, hLp,
          a, iF, iS, ap, iFp, iSp, d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp, xrf, xrs, xs1, xs2,
          xk1, Jrelnp, Jrelp, CaMKt]

    n_Statevariables = len(X0)

    # convert list to a numpy array
    X0 = array(X0)

    # cell type:  endo = 0, epi = 1, M = 2
    self.celltype = 1
    # The time over which to integrate
    self.t = sp.arange(0.0, 550.0, 0.01)
    # call the mdel with mode = 0 to return the state variables
    self.mode == 0

    beats = 3  # the number of beats in the simulation
    if SaveAll == True:
        XAll = np.empty([len(self.t), n_Statevariables, beats])

    for n in range(beats):
        beat_time = time.time()
        X1 = odeint(self.dALLdt, X0, self.t, args=(self,))
        if SaveAll == True:
            XAll[:, :, n] = X1

        # Extract the state variables at the end of the episode
        X0 = X1[-1, :]

        # print the beat number to monitor the runtime progress
        print('Action potential %s of %s (%.1fs)' % ((n + 1), beats, time.time() - beat_time))

    print('\nDone with simulations. The total execution time was %.1fs: ' % (time.time() - start_time))

    '''
    Save the data to disk (in numpy format) for later analysis
    '''
    if __debugging__:
        pdb.set_trace()

    if Savetodisk == True:
        if SaveAll == True:
            filename = 'XAll'
            np.save(filename, XAll)
        else:
            filename = 'X1'
            np.save(filename, X1)
        print('The simulated data were saved as %s.npy (a numpy array) for later analysis' % (filename))

    # Calculate currents and fluxes for the last beat (in X0)
    # Reduce the dataset since it takes quite some time
    calc_time = time.time()
    data_reduction = 10
    n_fluxes = 23

    print('\nCalculating currents and fluxes ...')
    t_new = self.t[::data_reduction].copy()
    X1_new = X1[::data_reduction, ].copy()
    n_points = len(t_new)

    IJ = np.empty([n_points, n_fluxes])
    self.mode = 1
    for i in range(n_points):
        IJ[i, :] = self.dALLdt(X1_new[i, :], t_new[i], self)
    print('Done with calculations. The total execution time was %.1fs: ' % (time.time() - calc_time))

    '''
    Plot the voltage and currents of the last beat
    '''
    print('\nPlotting data ...')
    V = X1[:, 0]
    INa = IJ[:, 0]
    INaL = IJ[:, 1]
    ICaL = IJ[:, 3]
    IKr = IJ[:, 4]

    plt.figure(figsize=(6, 8), dpi=96, facecolor='w', edgecolor='k')
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    plt.subplot(4, 1, 1)
    plt.title('O\'Hara Human ' + cell_type[self.celltype] + ' Ventricular AP')
    plt.plot(self.t, V, 'k')
    plt.ylabel('Voltage (mV)')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)

    plt.subplot(4, 1, 2)
    plt.plot(t_new, INa + INaL, 'b')
    plt.ylabel('$I_{Na} (pA)$')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)

    plt.subplot(4, 1, 3)
    plt.plot(t_new, ICaL, 'r')
    plt.ylabel('$I_{CaL} (pA)$')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)

    plt.subplot(4, 1, 4)
    plt.plot(t_new, IKr, 'g')
    plt.ylabel('$I_{Kr} (pA)$')
    plt.xlabel('Time (ms)')

    plt.show()
    print('Finished')


if __name__ == '__main__':
    runner = ohara()
    Main(runner)
