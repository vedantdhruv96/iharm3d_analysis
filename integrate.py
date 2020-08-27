#####################################################

# Functions to integrate and perform averages       #

#####################################################


import numpy as np


# Function to compute disk-avg quantities (as defined in Porth et al. (2019))
# Argument must be variable to avg., grid dict, disk boundary x2 indices
def disk_avg(var, grid, thmin, thmax):
    return (np.sum(np.sum(var[:,thmin:thmax,:]*grid['gdet'][:,thmin:thmax,:],axis=2)*grid['dx3'],axis=1)*grid['dx2'])/(np.sum(np.sum(grid['gdet'][:,thmin:thmax,:],axis=2)*grid['dx3'],axis=1)*grid['dx2'])


# Function to compute flux at r=r0, ie (x2,x3) integration
# Argument must be variable, grid dict, r0 index (r0 = radius at which flux is to be computed)
def shell_sum(var, grid, rind):
    return np.sum(np.sum(var[rind,:,:]*grid['gdet'][rind,:,:],axis=1)*grid['dx3'],axis=0)*grid['dx2']


# Function to compute numerator and denominator for density-weighted quantities (radial profiles)
# Argument must be variable, density, grid dict
def rho_weighted_frac(var, rho, grid):
    num = np.sum(np.sum(var*rho*grid['gdet'],axis=2)*grid['dx3'],axis=1)*grid['dx2']
    denom = np.sum(np.sum(rho*grid['gdet'],axis=2)*grid['dx3'],axis=1)*grid['dx2']
    return [num, denom]


# Function to perform volume sum
# Argument must be variable to be integrated, grid dict and radius (optional)
def volume_sum(var, grid, subdomain=False, rind=None):
    if subdomain:
        return np.sum(np.sum(np.sum(var[:rind,:,:]*grid['gdet'][:rind,:,:],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']
    else:
        return np.sum(np.sum(np.sum(var*grid['gdet'],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']


# Function to integrate in a given sector 
# Argument must be variable to be integrated, grid dict, sector boundaries (x2,x3) indices, radius (optional)
def sector_sum(var, grid, thmin, thmax, phimin, phimax, subdomain=False, rind=None):
    if subdomain:
        return np.sum(np.sum(np.sum(var[:rind,thmin:thmax,phimin:phimax]*grid['gdet'][:rind,thmin:thmax,phimin:phimax],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']
    else:
        return np.sum(np.sum(np.sum(var[:,thmin:thmax,phimin:phimax]*grid['gdet'][:,thmin:thmax,phimin:phimax],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']

# Function to compute theta profile at r=r0 (averages in phi). t-averaging will be done later
# Argument must be var, radius
def phi_avg_r(var, rind):
    return np.mean(var,axis=2)[rind,:]
