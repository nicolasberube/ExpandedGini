#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:40:53 2020

@author: berube
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def hill(classprop, q):
    """Hill-type diveristy measure

    Corresponds to the Leinster-Cobbold indicator when classes as perfectly
    dissimilar.

    Parameters
    ----------
    classprop: list of float
        The classes proportion of the element. The elements of
        the list should sum to 1.

    q: float
        The q parameter of the indicator, also known as the
        sensitivity parameter.
        Can be anything from ]0, +inf[, exclusively
    """

    if len(classprop) == 0:
        return -1

    if q == 1:
        som = 1
        for i in classprop:
            som /= np.power(i, i)
        return som
    else:
        som = 0
        for i in classprop:
            som += np.power(i, q)
        return np.power(som, 1/(1-q))


def plot_transfer(ax,
                  N,
                  X_res=100,
                  q_res=10):
    """Plot the Leinster-Cobbold (Hill-type) diversity for a class transfer

    Considering perfectly dissimilar classes, plots the diversity
    measure from a diversity of N (presence of equal proportion of N classes)
    to 1 (presence of a single class), for various values of the
    sensitivity parameter q.

    Parameters
    ----------
    ax: matplotlib subplot
        The subplot object to plot the graph on

    N: int
        Number of classes at one end of the transfer

    X_res: int, optional
        Resolution of the x-axis of the plot. Default is 1000

    q_res: int, optional
        Resolution of the span of the q parameter. Default is 500
    """
    q_values = [10**i for i in np.linspace(-2, 2, q_res)]
    q_values = q_values[::-1]
    #q_values = ([1] +
    #            q_values[:len(q_values)//2][::-1] +
    #            q_values[(len(q_values)+1)//2:])
    for iq, q in enumerate(q_values):
        color_idx = 1-iq/(len(q_values)-1)
        #if iq <= len(q_values)//2:
        #    color_idx = 0.5 - color_idx
        color = cm.gist_ncar(color_idx)

        X = np.linspace(0, 1, num=X_res)
        Y = []
        for x in X:
            classprop = [(1 + x*(N-1))/N] + [(1-x)/N]*(N-1)
            Y.append(hill(classprop, q))
        ax.plot(X, Y, color=color, linewidth=750/q_res)
    ax.plot([0, 1],
            [N, 1],
            ls='--',
            color='k')
    ax.set_title(f'Transfer from {N} to 1')


if __name__ == '__main__':
    q_res = 1000

    fig = plt.figure(constrained_layout=True)
    widths = [1, 1]
    heights = [5, 1]
    spec = fig.add_gridspec(ncols=2,
                            nrows=2,
                            width_ratios=widths,
                            height_ratios=heights)
    # Plots of class transfers
    ax1 = fig.add_subplot(spec[0, 0])
    plot_transfer(ax1, 2, q_res=q_res)

    ax2 = fig.add_subplot(spec[0, 1])
    plot_transfer(ax2, 50, q_res=q_res)

    # Legend
    legend = fig.add_subplot(spec[1, :])
    legend.set_title('q value')
    q_values = [10**i for i in np.linspace(-2, 2, q_res)]
    for iq, q in enumerate(q_values):
        color_idx = iq/(len(q_values)-1)
        color = cm.gist_ncar(color_idx)
        legend.semilogx([q, q],
                        [0, 1],
                        color=color,
                        linewidth=750/q_res)
        legend.get_yaxis().set_visible(False)
    plt.savefig('transfer.png', dpi=600)
