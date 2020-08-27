#############################################################

# Calculates fluxes, disk-avg. quantities, jet power, ...   #
# NOTE: Edit model_dir location                            #

#############################################################


import numpy as np
import read_data as read
import integrate
import quantities as quant
import h5py
import plot as plot
import os, sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import parallelize as par


# Call script as python compute.py padding path/to/dumps dstart dend
# padding: determines number of processes to spawn (0<pad<1)
# On BH: 0.6 for SANE and 0.25 for MAD
# path/to/dumps: dumps location wrt where you're running script from
# dstart, dend: starting and ending dump file numbers for analysis


# Initialization: Obtaining and storing time range, outer boundary and disk boundaries for analysis, loading grid dict, creating requisite directories
print("------Beginning analysis------")
pad = float(sys.argv[1])
dumps_loc = sys.argv[2]
dstart = int(sys.argv[3])
dend = int(sys.argv[4])
start_dump = read.load_dump_basic(os.path.join(dumps_loc,'dump_0000{0:04}.h5'.format(dstart)))
tstart = start_dump['t']
#model = start_dump['model'] ###REMOVE FOR FRONTERA RUNS
model = 'SANE'  ### EDIT
spin = start_dump['a']
tend = read.load_dump_basic(os.path.join(dumps_loc,'dump_0000{0:04}.h5'.format(dend)))['t']
grid_loc = os.path.join(dumps_loc,'grid.h5')
grid = read.load_grid(os.path.join(dumps_loc,'dump_0000{0:04}.h5'.format(dstart)),grid_loc)
print("------Grid file loaded------")
resolution = str(grid['n1'])+'x'+str(grid['n2'])+'x'+str(grid['n3'])
if model == 'SANE':
    r_out = 40
elif model == 'MAD':
    r_out = 100
r_out_ind = np.argmax(grid['r'][:,0,0]-r_out>0)-1
rEH_ind = np.argmax(grid['r'][:,0,0]-grid['rEH']>0)-1
thmin = np.pi/3.
thmax = 2*np.pi/3.
thmin_ind = np.argmin(abs(grid['th'][-1,:,0]-thmin))
thmax_ind = np.argmin(abs(grid['th'][-1,:,0]-thmax))

print("Start time: {0:.2f}\nEnd time: {1:.2f}".format(tstart, tend))

dumps_list = range(dstart, dend+1)

model_dir = os.path.join('/fs1/vdhruv/eht_IHARM',model) ###CHANGE FOR FRONTERA RUNS
output_dir = os.path.join(model_dir,str(spin),resolution)
flux_dir = os.path.join(output_dir,'flux_stuff')
disk_avg_dir = os.path.join(output_dir,'disk_avg_stuff')
scale_dir = os.path.join(output_dir,'scale_height_stuff')
jp_dir = os.path.join(output_dir,'jet_power_stuff')
omega_dir = os.path.join(output_dir,'rotational_stuff')
try:
    os.makedirs(output_dir)
    os.makedirs(flux_dir)
    os.makedirs(disk_avg_dir)
    os.makedirs(scale_dir)
    os.makedirs(jp_dir)
    os.makedirs(omega_dir)
except OSError:
    print("Data directory already exists")
    pass
if os.path.isfile(os.path.join(model_dir,'radii_'+resolution+'.h5')):
    try:
        hfp = h5py.File(os.path.join(model_dir,'radii_'+resolution+'.h5'),'a')
        hfp[model+str(spin)] = grid['r'][:,0,0]
        hfp.close()
    except RuntimeError:
        print("Radii already written")
        pass
else:
    hfp = h5py.File(os.path.join(model_dir,'radii_'+resolution+'.h5'),'w')
    hfp[model+str(spin)] = grid['r'][:,0,0]
    hfp.close()

hfp = h5py.File(os.path.join(output_dir,'misc.h5'),'w')
hfp['tstart'] = tstart
hfp['tend'] = tend
hfp.close()


# Function to perform required analysis
def calculation(dump_num):
    dump = read.load_dump_all(os.path.join(dumps_loc,'dump_0000{0:04}.h5'.format(dump_num)), grid)
    print("Analyzing dump {0:04d}".format(dump_num))
    
    # flux calculation
    mdot = -integrate.shell_sum(dump['RHO']*dump['ucon'][Ellipsis,1], grid, rEH_ind)
    phi_bh = 0.5*integrate.shell_sum(abs(dump['B'][Ellipsis,0]), grid, rEH_ind)
    ldot = integrate.shell_sum(quant.Tmixed(dump, i=1, j=3), grid, rEH_ind)
    fout = integrate.shell_sum(-quant.Tmixed(dump, i=1, j=0), grid, rEH_ind)
    hfp = h5py.File(os.path.join(flux_dir,'flux_values_{0:04}.h5'.format(dump_num)),'w')
    hfp['mdot'] = mdot
    hfp['phi_bh'] = phi_bh
    hfp['ldot'] = ldot
    hfp['fout'] = fout
    hfp.close()

    # disk-average calculation: generating radial profiles
    rho_avg = integrate.disk_avg(dump['RHO'], grid, thmin_ind, thmax_ind)
    pg_avg = integrate.disk_avg(dump['pg'], grid, thmin_ind, thmax_ind)
    B_avg = integrate.disk_avg(np.sqrt(dump['bsq']), grid, thmin_ind, thmax_ind)
    B_sq_avg = integrate.disk_avg(dump['bsq'], grid, thmin_ind, thmax_ind)
    sigma_avg = B_sq_avg/rho_avg
    beta_inv_avg = 0.5*(B_sq_avg/pg_avg)
    hfp = h5py.File(os.path.join(disk_avg_dir,'disk_avg_radial_profiles_{0:04}.h5'.format(dump_num)),'w')
    hfp['rho_avg'] = rho_avg
    hfp['pg_avg'] = pg_avg
    hfp['B_avg'] = B_avg
    hfp['sigma_avg'] = sigma_avg
    hfp['beta_inv_avg'] = beta_inv_avg
    hfp.close()

    # scale-height calculation (computed as Eq (21) in Porth et al. (2019))
    fraction = integrate.rho_weighted_frac(abs(np.pi/2-grid['th']), dump['RHO'], grid)
    hfp = h5py.File(os.path.join(scale_dir,'scale_height_frac_values_{0:04}.h5'.format(dump_num)),'w')
    hfp['num'] = fraction[0]
    hfp['denom'] = fraction[1]
    hfp.close()

    #jet power calculation
    fdiff = (np.sum((-quant.Tmixed(dump, i=1, j=0) - dump['RHO']*dump['ucon'][Ellipsis,1])*grid['gdet'], axis=2)*grid['dx3'])[r_out_ind,:]
    fout = np.mean(-quant.Tmixed(dump, i=1, j=0), axis=2)[r_out_ind,:]
    fm = np.mean(dump['RHO']*dump['ucon'][Ellipsis,1], axis=2)[r_out_ind,:]
    mdot = -integrate.shell_sum(dump['RHO']*dump['ucon'][Ellipsis,1], grid, rEH_ind)
    rho = np.mean(dump['RHO'], axis=2)[r_out_ind, :]
    bsq = np.mean(dump['bsq'], axis=2)[r_out_ind, :]
    hfp = h5py.File(os.path.join(jp_dir,'jet_power_related_values_{0:04}.h5'.format(dump_num)),'w')
    hfp['rho'] = rho
    hfp['bsq'] = bsq
    hfp['fdiff'] = fdiff
    hfp['fout'] = fout
    hfp['fm'] = fm
    hfp['mdot'] = mdot
    hfp.close()

    #rotational profile calculation
    fraction = integrate.rho_weighted_frac(dump['ucon'][Ellipsis,3]/dump['ucon'][Ellipsis,0], dump['RHO'], grid)
    hfp = h5py.File(os.path.join(omega_dir,'omega_frac_values_{0:04}.h5'.format(dump_num)),'w')
    hfp['num'] = fraction[0]
    hfp['denom'] = fraction[1]
    hfp.close()


print("------Looping over dump files------")
Nthreads = par.calc_threads(pad)
print("Number of threads: {0:02d}".format(Nthreads))
par.run_parallel(calculation, dumps_list, Nthreads)
print("------Looping complete------")
