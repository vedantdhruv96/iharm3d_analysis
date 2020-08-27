##########################################################

# Load fluid and grid data into dictionaries             #

##########################################################


import numpy as np
import h5py
import os
import coordinates
import quantities as quant


# Load basic info from dump file. Additional fluid info will be calculated once grid is loaded
# Argument must be path to dump
def load_dump_basic(dump_loc):
    #print("-----Loading basic info-----")
    dfile = h5py.File(dump_loc,'r')
    dump = {}
    dump['t'] = dfile['t'][()]
    dump['gamma'] = dfile['header/gam'][()]
    try:
        dump['rEH'] = dfile['header/geom/mmks/Reh'][()]
        #print("-----header/geom/mmks/Reh is a valid key-----")
    except KeyError:
        pass#print("-----header/geom/mmks/Reh not a valid key-----")
    try:
        dump['rEH'] = dfile['header/geom/mmks/r_eh']
        #print("-----header/geom/mmks/r_eh is a valid key-----")
    except KeyError:
        pass#print("-----header/geom/mmks/r_eh not a valid key-----")
    dump['a'] = dfile['header/geom/mmks/a'][()]
    """dump['prob'] = dfile['header/problem/PROB'][()]
    if dfile['header/problem/mad_type'][()]==0:
        dump['model'] = 'SANE'
    else:
        dump['model'] = 'MAD'
    dump['rin'] = dfile['header/problem/rin'][()]
    dump['rmax'] = dfile['header/problem/rmax'][()]"""
    dump['n_prim'] = dfile['header/n_prim'][()]
    dump['prim_list'] = dfile['header/prim_names'][()]
    dump['jcon'] = dfile['jcon'][()]
    dump['RHO'] = dfile['prims'][()][Ellipsis,0]
    dump['UU'] = dfile['prims'][()][Ellipsis,1]
    dump['U'] = dfile['prims'][()][Ellipsis,2:5]
    dump['B'] = dfile['prims'][()][Ellipsis,5:8]
    dfile.close()
    return dump


# Load grid (geometry) data. Useful to compute additional fluid state data
# Argument must be path to a dump file and grid.h5
def load_grid(dump_loc, grid_loc):
    #print("-----Loading grid-----")
    dfile = h5py.File(dump_loc,'r')
    gfile = h5py.File(grid_loc,'r')
    grid = {}
    grid['n1'] = dfile['/header/n1'][()]; grid['n2'] = dfile['/header/n2'][()]; grid['n3'] = dfile['/header/n3'][()]
    grid['a'] = dfile['header/geom/mmks/a'][()]
    try:
        grid['rEH'] = dfile['header/geom/mmks/Reh'][()]
    except KeyError:
        pass
    try:
        grid['rEH'] = dfile['header/geom/mmks/r_eh'][()]
    except KeyError:
        pass
    grid['hslope'] = dfile['header/geom/mmks/hslope'][()]
    grid['mks_smooth'] = dfile['header/geom/mmks/mks_smooth'][()]
    grid['poly_alpha'] = dfile['header/geom/mmks/poly_alpha'][()]
    grid['poly_xt'] = dfile['header/geom/mmks/poly_xt'][()]
    grid['dx1'] = dfile['/header/geom/dx1'][()]; grid['dx2'] = dfile['/header/geom/dx2'][()]; grid['dx3'] = dfile['/header/geom/dx3'][()]
    grid['D'] = (np.pi*grid['poly_xt']**grid['poly_alpha'])/(2*grid['poly_xt']**grid['poly_alpha']+(2/(1+grid['poly_alpha'])))
    grid['startx1'] = dfile['header/geom/startx1'][()]; grid['startx2'] = dfile['header/geom/startx2'][()]; grid['startx3'] = dfile['header/geom/startx3'][()];
    grid['x1'] = gfile['X1'][()]; grid['x2'] = gfile['X2'][()]; grid['x3'] = gfile['X3'][()]
    grid['r'] = gfile['r'][()]; grid['th'] = gfile['th'][()]; grid['phi'] = gfile['phi'][()]
    grid['x'] = gfile['X'][()]; grid['y'] = gfile['Y'][()]; grid['z'] = gfile['Z'][()] 
    grid['gdet'] = gfile['gdet'][()]
    grid['gcon'] = gfile['gcon'][()]
    grid['gcov'] = gfile['gcov'][()]
    grid['lapse'] = gfile['lapse'][()]
    dfile.close(); gfile.close()
    return grid


# Load (essentially compute) more fluid state data such as 4-velocities, magnetic field 4-vector (and some more)
# Returns dump.
# Argument must be path to dump file and grid dict
def load_dump_all(dump_loc, grid):
    dump = load_dump_basic(dump_loc)
    #print("-----Loading more fluid state data-----")
    dump['ucon'], dump['ucov'], dump['bcon'], dump['bcov'] = quant.compute_u_b(dump, grid)
    dump['pg'] = (dump['gamma']-1)*dump['UU']
    dump['bsq'] = np.einsum('ijkm,ijkm->ijk',dump['bcon'],dump['bcov'])
    dump['beta'] = 2*dump['pg']/dump['bsq']
    dump['sigma'] = dump['bsq']/dump['RHO']
    return dump
