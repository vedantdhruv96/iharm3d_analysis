##################################################################

# Toroidal, poloidal, time series, radial profiles and more      #

##################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


# Function to generate poloidal (x,z) slice
# Argument must be variable, patch pole (to have x coordinate plotted correctly), averaging in phi option
def xz_slice(var, patch_pole=False, average=False):
    n1 = var.shape[0]; n2 = var.shape[1]; n3 = var.shape[2]
    xz_var = np.zeros((2*n1,n2))
    if average:
        var = np.mean(var,axis=2)
        for i in range(n1):
            xz_var[i,:] = var[n1-1-i,:]
            xz_var[i+n1,:] = var[i,:]
    else:
        for i in range(n1):
            xz_var[i,:] = var[n1-1-i,:,n3//2]
            xz_var[i+n1,:] = var[i,:,0]
    if patch_pole:
        xz_var[:,0] = xz_var[:,-1] = 0
    return xz_var


# Function to generate toroidal (x,y) slice
# Argument must be variable, averaging in theta option
def xy_slice(var, average=False, patch_phi=False):
    n1 = var.shape[0]; n2 = var.shape[1]; n3 = var.shape[2]
    if average:
        xy_var = np.mean(var,axis=1)
    else:
        xy_var = var[:,n2//2,:]
    #xy_var = np.vstack((xy_var.transpose(),xy_var.transpose()[0])).transpose()
    if patch_phi:
        xy_var[:,0] = xy_var[:,-1] = 0
    return xy_var


# Function to generate poloidal plot
# Argument must be variable, grid dict, cmap, vmin, vmax, domain, bh, half cut, midplane folded quadrant, shading (all but first two are default arguments)
#CAVEAT: If plotting halfcut with timestamp, edit textbox location accordingly
def poloidal(var, grid, title=False, timestamp=False, cmap='jet', vmin=None, vmax=None, domain = [-30,30,-30,30], bh=True, halfcut=False, midfold=False, shading='gouraud'):
    ax = plt.gca()
    n1 = grid['n1']; n2 = grid['n2']; n3 = grid['n3']
    if midfold:
        x = xz_slice(grid['x'], patch_pole=True)[n1:,:n2//2]
        z = xz_slice(grid['z'])[n1:,:n2//2]
        var = xz_slice(var, average=True)[n1:,:]
        var = (var[:,:n2//2]+np.flip(var[:,n2//2:],axis=1))/2
        domain[0]=0; domain[2]=0
    elif halfcut:
        x = xz_slice(grid['x'], patch_pole=True)[n1:,:]
        z = xz_slice(grid['z'])[n1:,:]
        var = xz_slice(var, average=True)[n1:,:]
        domain[0]=0
    else:
        x = xz_slice(grid['x'], patch_pole=True)
        z = xz_slice(grid['z'])
        var = xz_slice(var, average=True)
    polplot = ax.pcolormesh(x, z, var, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax.set_xlabel('$x (GM/c^2)$')
    ax.set_ylabel('$z (GM/c^2)$')
    ax.set_xlim(domain[:2])
    ax.set_ylim(domain[2:])
    if bh:
        circle = plt.Circle((0,0),grid['rEH'],color='k')
        ax.add_artist(circle)
    ax.set_aspect('equal')
    if timestamp:
        box = dict(boxstyle='square',facecolor='white')
        ax.text(0.9,0.95,'$t=$'+timestamp, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, bbox=box)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(polplot, cax=cax)
    if title:
        ax.set_title(title)
    return ax


# Function to generate toroidal plot
# Argument must be variable, grid dict, cmap, vmin, vmax, domain, bh, half cut, midplane folded quadrant, shading (all but first two are default arguments)
def toroidal(var, grid, title=False, timestamp=False, cmap='jet', vmin=None, vmax=None, domain = [-50,50,-50,50], bh=True, shading='gouraud'):
    ax = plt.gca()
    n1 = grid['n1']; n2 = grid['n2']; n3 = grid['n3']
    x = xy_slice(grid['x'])
    y = xy_slice(grid['y'], patch_phi=True)
    var = xy_slice(var, average=True)
    torplot = ax.pcolormesh(x, y, var, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax.set_xlabel('$x (GM/c^2)$')
    ax.set_ylabel('$y (GM/c^2)$')
    ax.set_xlim(domain[:2])
    ax.set_ylim(domain[2:])
    if bh:
        circle = plt.Circle((0,0),grid['rEH'],color='k')
        ax.add_artist(circle)
    ax.set_aspect('equal')
    if timestamp:
        box = dict(boxstyle='square',facecolor='white')
        ax.text(0.9,0.95,'$t=$'+timestamp, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, bbox=box)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(torplot, cax=cax)
    if title:
        ax.set_title(title)
    return ax


# Function to generate time series for fluxes (accretion rate, magnetic flux, ang. momentum flux, energy flux
# Argument must be time series, flux arrays, title (optional)
def flux_series(t, mdot, phi_bh, ldot, fdiff):
    nrows = 2; ncols =2
    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows,ncols)
    fig.set_size_inches(14,8)
        
    ax0.plot(t, np.abs(mdot))
    ax0.set_xlabel('$t(GM/c^3)$')
    ax0.set_ylabel('$\\vert\\dot{M}\\vert$')
     
    ax1.plot(t, phi_bh/np.sqrt(np.abs(mdot)))
    ax1.set_xlabel('$t(GM/c^3)$')
    ax1.set_ylabel('$\\frac{\\Phi_{BH}}{\\sqrt{\\vert\\dot{M}\\vert}}$')

    ax2.plot(t, np.abs(ldot)/np.abs(mdot))
    ax2.set_xlabel('$t(GM/c^3)$')
    ax2.set_ylabel('$\\frac{\\vert\\dot{L}\\vert}{\\vert\\dot{M}\\vert}$')

    ax3.plot(t, np.abs(fdiff)/np.abs(mdot))
    ax3.set_xlabel('$t(GM/c^3)$')
    ax3.set_ylabel('$\\frac{\\vert\\dot{E}-\\dot{M}\\vert}{\\vert\\dot{M}\\vert}$')

    plt.tight_layout()

    return fig


# Fucntion to generate disk-averaged radial profiles (density, gas pressure, magnetic field strength, magnetization, plasma beta inv.)
# Argument must be radii dict {"model":radii values}, data dict {"model": {"variable":values}}, models list, variables list
def disk_avg_plot(radii, data, models, variables):
    fig = plt.gcf()
    nrows=2; ncols=3
    gs = gridspec.GridSpec(nrows,ncols)
    ctr1=0; ctr2=0
    for var in variables:
        ax = fig.add_subplot(gs[ctr1,ctr2])
        for model in sorted(models):
            r = radii[str(model)]; rmax=50; rmax_ind = np.argmax(r-rmax>0)-1
            ax.loglog(r[:rmax_ind],data[str(model)][var][:rmax_ind],label=str(model))
            ax.set_xlabel('$r(GM/c^2)$',size=14)
            if var == 'rho':
                ax.set_ylabel('$\\langle\\rho\\rangle$',size=14)
            if var == 'pg':
                ax.set_ylabel('$\\langle p_{gas}\\rangle$',size=14)
            if var == 'B':
                ax.set_ylabel('$\\langle B\\rangle$',size=14)
            if var == 'sigma':
                ax.set_ylabel('$\\langle\\sigma\\rangle$',size=14)
            if var == 'beta_inv':
                ax.set_ylabel('$\\langle\\beta^{-1}\\rangle$',size=14)
            ax.tick_params(axis='both',which='major',labelsize=12)
            ax.grid(True)
            if (ctr1==0 and ctr2==0):
                ax.legend(loc='lower right')
        ctr2+=1
        if ctr2>=ncols:
            ctr1+=1
            ctr2=0
    plt.tight_layout()
    return fig


# Function to plot density scale height (computed as Eq (21) in Porth et al. (2019))
# Argument must be radii dict {"model":radii values}, data dict{"model": {"variable":values}} which are numerator and denominator values, models list
def scale_height(radii, data, models):
    ax = plt.gca()
    for model in models:
        r = radii[str(model)]; rmax=50; rmax_ind = np.argmax(r-rmax>0)-1
        ax.plot(r[:rmax_ind],data[str(model)]['num'][:rmax_ind]/data[str(model)]['denom'][:rmax_ind],label=str(model))
        ax.set_xlabel('$r(GM/c^2)$',size=14)
        ax.set_ylabel('$H/R(r)$',size=14)
        ax.tick_params(axis='both',which='major',labelsize=12)
        ax.grid(True)
    ax.legend(loc='best')
    return ax


# Function to plot theta-profiles of density, magnetic field strength^2, energy flux, betagamma at outer boundary
# Argument must be theta profiles of above quantities, grid dict, outer radius index
def theta_profiles(rho, bsq, fdiff, bg, grid, r_out_ind):
    theta = grid['th'][r_out_ind,:,0]; n2 = grid['n2']
    
    fig = plt.gcf()
    nrows=2; ncols=2
    gs = gridspec.GridSpec(nrows,ncols)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.semilogy(theta[:n2//2], rho[:n2//2],color='green')
    ax0.set_xlabel('$\\theta$')
    ax0.set_ylabel('$\\rho(\\theta)$')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(theta[:n2//2], bsq[:n2//2],color='green')
    ax1.set_xlabel('$\\theta$')
    ax1.set_ylabel('$b^2(\\theta)$')

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(theta[:n2//2], fdiff[:n2//2],color='green')
    ax2.set_xlabel('$\\theta$')
    ax2.set_ylabel('$(-T^r_t-\\rho u^r)(\\theta)$')

    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(theta[:n2//2], bg[:n2//2],color='green')
    ax3.set_xlabel('$\\theta$')
    ax3.set_ylabel('$\\beta\\gamma(\\theta)$')
    
    plt.tight_layout()
    return fig

# Function to plot density-weighted rotational profile and possible fits
# Argument must be omega(r), grid dict, max. radius, fits (optional)
def rotational_profile(omega, grid, rmax, kepfit=False, omegafit=False):
    a = grid['a']; rEH = grid['rEH']; r = grid['r'][:,0,0]
    rmax_ind = np.argmax(r-rmax>0)-1
    
    fig = plt.gcf()
    fig.set_size_inches(11,7)
    plt.loglog(r[:rmax_ind], omega[:rmax_ind],label='Data')
    if kepfit.any():
        plt.loglog(r[:rmax_ind], kepfit[:rmax_ind], ls='dashed', label='Keplerian fit')
    if omegafit.any():
        plt.loglog(r[:rmax_ind], omegafit[:rmax_ind], label='Fit')
    plt.gca().axvline(rEH, ls='dashdot',color='k')
    plt.xlabel('$r$')
    plt.ylabel('$\\langle\\Omega\\rangle_{\\rho}$')
    plt.legend()
    return fig
