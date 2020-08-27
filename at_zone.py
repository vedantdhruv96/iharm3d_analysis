################################################################

# Compute fluid quantities in quantities.py at a single zone   # 
# Useful for comparing codes                                  # 

################################################################


import numpy as np
import os,sys
import read_data as read
import quantities as quant


### Call script as: python at_zone.py /path/to/dumps/ dump_number zone_indices quantity component

### Dump number must be 4 digits, zero padded

### Zone indices must be space separated, ensure that the zone indices are within the grid of the model

### Input for quantities:
### Contra- and co- variant metric, 4-velocity, mag. field 4-vector ============> ub
### Contravariant MHD stress-energy tensor =====================================> tcon
### Covariant MHD stress-energy tensor =========================================> tcov
### Mixed MHD stress-eenrgy tensor =============================================> tmixed
### Mixed EM stress-energy tensor ==============================================> tem_mixed
### Mixed fluid stress-energy tensor ===========================================> tfl_mixed
### Contravariant Maxwell stress tensor ========================================> fcon
### Covariant Maxwell stress tensor ============================================> fcov

### Components must be provided if quantity is neq 'ub'


path = sys.argv[1]
dump = sys.argv[2]
zone = [int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
quantity = sys.argv[6]
dump_loc = os.path.join(path,"dump_0000"+dump+".h5")
grid_loc = os.path.join(path,"grid.h5")
if len(sys.argv)>7:
    i = int(sys.argv[7])
    j = int(sys.argv[8])


if quantity == 'ub':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid) 
    print(quant.compute_u_b(dump,grid,atzone=True,zone=zone))

elif quantity == 'tcon':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Tcon(dump,grid,atzone=True,zone=zone,i=i,j=j))

elif quantity == 'tcov':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Tcov(dump,grid,atzone=True,zone=zone,i=i,j=j))

elif quantity == 'tmixed':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Tmixed(dump,grid,atzone=True,zone=zone,i=i,j=j))

elif quantity == 'tem_mixed':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Tmixed_EM(dump,grid,atzone=True,zone=zone,i=i,j=j))

elif quantity == 'tfl_mixed':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Tmixed_Fl(dump,grid,atzone=True,zone=zone,i=i,j=j))

elif quantity == 'fcon':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Fcon(dump,grid,atzone=True,zone=zone,i=i,j=j))

elif quantity == 'fcov':
    grid = read.load_grid(dump_loc, grid_loc)
    dump = read.load_dump_all(dump_loc, grid)
    print(quant.Fcov(dump,grid,atzone=True,zone=zone,i=i,j=j))

