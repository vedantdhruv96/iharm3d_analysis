##########################################################

# Compute transformation matrix                          #
# Kerr-Schild (KS) -> Funky Modified Kerr-Schild (FMKS)  #
# FMKS -> KS                                             #
# KS -> Boyer-Lindquist (BL)                             #

#########################################################
 

import numpy as np


# Transformation matrix for KS->FMKS
# Argument must be grid dict from read_data.py
def dxdX_KS_to_FMKS(grid):
    dxdX = np.zeros((grid['n1'],grid['n2'],grid['n3'],4,4),dtype=float)
    theta_g = (np.pi*grid['x2'])+((1-grid['hslope'])/2)*(np.sin(2*np.pi*grid['x2']))
    theta_j = grid['D']*(2*grid['x2']-1)*(1+(((2*grid['x2']-1)/grid['poly_xt'])**grid['poly_alpha'])/(1+grid['poly_alpha'])) + np.pi/2
    derv_theta_g = np.pi + (1-grid['hslope'])*np.pi*np.cos(2*np.pi*grid['x2'])
    derv_theta_j = (2*grid['poly_alpha']*grid['D']*(2*grid['x2']-1)*((2*grid['x2']-1)/grid['poly_xt'])**(grid['poly_alpha']-1))/(grid['poly_xt']*(grid['poly_alpha']+1)) + 2*grid['D']*(1 + (((2*grid['x2']-1)/grid['poly_xt'])**grid['poly_alpha'])/(grid['poly_alpha']+1))
    dxdX[Ellipsis,0,0] = dxdX[Ellipsis,3,3] = 1
    dxdX[Ellipsis,1,1] = np.exp(grid['x1'])
    dxdX[Ellipsis,2,1] = -grid['mks_smooth']*np.exp(-grid['mks_smooth']*grid['Dx1'][:,np.newaxis,np.newaxis])*(theta_j-theta_g)
    dxdX[Ellipsis,2,2] = derv_theta_g+np.exp(-grid['mks_smooth']*grid['Dx1'][:,np.newaxis,np.newaxis])*(derv_theta_j-derv_theta_g)
    return dxdX


# Transformation matrix for FMKS->KS
# Argument must be grid dict from read_data.py
def dxdX_FMKS_to_KS(grid):
    return (np.linalg.inv(dxdX_KS_to_FMKS(grid)))


# Transformation matrix for KS->BL
# Argument must be grid dict from read_data.py
def dxdX_KS_to_BL(grid):
    dxdX = np.zeros((grid['n1'],grid['n2'],grid['n3'],4,4),dtype=float)
    dxdX[Ellipsis,0,0] = dxdX[Ellipsis,1,1] = dxdX[Ellipsis,2,2] = dxdX[Ellipsis,3,3] = 1
    dxdX[Ellipsis,0,1] = -(2*grid['r'])/(grid['r']**2-2*grid['r']+grid['a']**2)
    dxdX[Ellipsis,3,1] = grid['a']/(grid['r']**2-2*grid['r']+grid['a']**2)
    return dxdX
