#########################################################################
 
# Compute fluid state quantities                                        #
# (With functionality to compute at single zone; call from at_zone.py)  #

#########################################################################


import numpy as np


# Function to calculate 4-velocity and magnetic field 4-vector
# Argument must be the dump and grid dicts from read_data.py. Option to enter zone indices if one wants value at single zone
def compute_u_b(dump, grid, atzone=False, zone=None):
    gti = grid['gcon'][Ellipsis,0,1:4]
    gij = grid['gcov'][Ellipsis,1:4,1:4]
    beta_i = np.einsum('ijks,ijk->ijks',gti,grid['lapse']**2)
    qsq = np.einsum('ijky,ijky->ijk',np.einsum('ijkxy,ijkx->ijky',gij,dump['U']),dump['U'])
    gamma = np.sqrt(1+qsq)
    ui = dump['U']-np.einsum('ijks,ijk->ijks',beta_i,gamma/grid['lapse'])
    ut = gamma/grid['lapse']
    ucon = np.append(ut[Ellipsis,None],ui,axis=3)
    ucov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],ucon)
    bt = np.einsum('ijkm,ijkm->ijk',np.einsum('ijksm,ijks->ijkm',grid['gcov'][Ellipsis,1:4,:],dump['B']),ucon)
    bi = (dump['B']+np.einsum('ijks,ijk->ijks',ui,bt))/ut[Ellipsis,None]
    bcon = np.append(bt[Ellipsis,None],bi,axis=3)
    bcov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],bcon)
    if atzone:
        return grid['gcon'][zone[0],zone[1],zone[2],:,:], grid['gcov'][zone[0],zone[1],zone[2],:,:], ucon[zone[0],zone[1],zone[2],:], ucov[zone[0],zone[1],zone[2],:], bcon[zone[0],zone[1],zone[2],:], bcov[zone[0],zone[1],zone[2],:]
    else:
        return ucon, ucov, bcon, bcov


# Functions to compute components of MHD, EM and fluid stress-energy tensor, and Faraday tensor
# Argument must be the dump and grid dicts from read_data.py. Option to enter zone indices if one wants value at single zone. NOTE: grid dict must not be passed if called mixed stress-energy tensor
def Tcon(dump, grid, atzone=False, zone=None, i=0, j=0):
    if atzone:
        return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucon'])) + np.einsum('ijk,ijkmn->ijkmn',dump['pg']+dump['bsq']/2,grid['gcon']) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcon']))[zone[0],zone[1],zone[2],i,j]
    else:
        return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucon'])) + np.einsum('ijk,ijkmn->ijkmn',dump['pg']+dump['bsq']/2,grid['gcon']) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcon']))[Ellipsis,i,j]


def Tcov(dump, grid, atzone=False, zone=None, i=0, j=0):
    if atzone:
        return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucov'],dump['ucov'])) + np.einsum('ijk,ijkmn->ijkmn',dump['pg']+dump['bsq']/2,grid['gcov']) - np.einsum('ijkm,ijkn->ijkmn',dump['bcov'],dump['bcov']))[zone[0],zone[1],zone[2],i,j]
    else:
        return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucov'],dump['ucov'])) + np.einsum('ijk,ijkmn->ijkmn',dump['pg']+dump['bsq']/2,grid['gcov']) - np.einsum('ijkm,ijkn->ijkmn',dump['bcov'],dump['bcov']))[Ellipsis,i,j]


def Tmixed(dump, atzone=False, zone=None, i=0, j=0):
    if atzone:
        if i==j:
            return ((np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[zone[0],zone[1],zone[2],i,j] + (dump['pg']+dump['bsq']/2)[zone[0],zone[1],zone[2],i,j])
        else:
            return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[zone[0],zone[1],zone[2],i,j]
    else:
        if i==j:
            return ((np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j] + (dump['pg']+dump['bsq']/2)[Ellipsis,i,j])
        else:
            return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j]


def Tmixed_EM(dump, atzone=False, zone=None, i=0, j=0):
    if atzone:
        if i==j:
            return ((np.einsum('ijk,ijkmn->ijkmn',dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[zone[0],zone[1],zone[2],i,j] + (dump['bsq']/2)[zone[0],zone[1],zone[2],i,j])
        else:
            return (np.einsum('ijk,ijkmn->ijkmn',dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[zone[0],zone[1],zone[2],i,j]
    else:
        if i==j:
            return ((np.einsum('ijk,ijkmn->ijkmn',dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j] + (dump['bsq']/2)[Ellipsis,i,j])
        else:
            return (np.einsum('ijk,ijkmn->ijkmn',dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j]


def Tmixed_Fl(dump, atzone=False, zone=None, i=0, j=0):
    if atzone:
        if i==j:
            return ((np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])))[zone[0],zone[1],zone[2],i,j] + (dump['pg'])[zone[0],zone[1],zone[2],i,j])
        else:
            return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])))[zone[0],zone[1],zone[2],i,j]
    else:
        if i==j:
            return ((np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])))[Ellipsis,i,j] + (dump['pg'])[Ellipsis,i,j])
        else:
            return (np.einsum('ijk,ijkmn->ijkmn',dump['RHO']+dump['UU']+dump['pg'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])))[Ellipsis,i,j]
