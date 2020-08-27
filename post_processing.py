#############################################################################

# Time-averages, plotting, creating post-proc. files, final calculations    #
# Run after running all models of interest                                  #
# NOTE: Edit data directory, model, resolution, spins and dumps location    #

#############################################################################


import numpy as np
import h5py
import os
import read_data as read
import matplotlib.pyplot as plt
import plot as plot
import parallelize as par
from decimal import Decimal


data_dir = ''
model = 'SANE'
res = '288x128x128'
spins = [-0.9375, -0.5, 0.0, 0.5, 0.9375]
dumps_locs = ['/bd3/eht/GRMHD/SANE/a-0.94/288x128x128_IHARM/dumps/','/bd3/eht/GRMHD/SANE/a-0.5/288x128x128_IHARM/dumps/','/bd3/eht/GRMHD/SANE/a0/288x128x128_IHARM/dumps/','/bd3/eht/GRMHD/SANE/a+0.5/288x128x128_IHARM/dumps/','/bd3/eht/GRMHD/SANE/a+0.94/288x128x128_IHARM/dumps/']; 
ctr = 0 #counter to keep track of dumps_loc list index
print("------Beginning analysis: %s------"%(model))
for spin in sorted(spins):
    model_dir = os.path.join(data_dir,model,str(spin),res)
    
    # Obtaining grid, dumps, times, radii and flux averaging
    flux_dir = os.path.join(model_dir,'flux_stuff')
    dstart = int(sorted(os.listdir(flux_dir))[0][-7:-3])
    dend = int(sorted(os.listdir(flux_dir))[-1][-7:-3])
    Ndumps = dend-dstart+1
    grid_loc = os.path.join(dumps_locs[ctr],'grid.h5')
    grid = read.load_grid(os.path.join(dumps_locs[ctr],'dump_0000{0:04}.h5'.format(dstart)),grid_loc)
    print("\n-------------------------------------------------------------")
    print('\nGrid file for ({0:s} {1:3.2f}) loaded '.format(model,spin))
    ctr+=1
    hfp = h5py.File(os.path.join(model_dir,'misc.h5'),'r')
    tstart = hfp['tstart'][()]
    tend = hfp['tend'][()]
    t = np.linspace(tstart, tend, Ndumps)
    hfp.close()
    hfp = h5py.File(os.path.join(data_dir,model,'radii_'+res+'.h5'),'r')
    r = hfp[model+str(spin)][()]
    hfp.close()
    
    mdot = np.zeros((Ndumps),dtype=float)
    phi_bh = np.zeros_like(mdot)
    ldot = np.zeros_like(mdot)
    fout = np.zeros_like(mdot)
    for i in range(dstart,dend+1):
        hfp = h5py.File(os.path.join(flux_dir,'flux_values_{0:04d}.h5'.format(i)),'r')
        mdot[i-dstart] = hfp['mdot'][()]
        phi_bh[i-dstart] = hfp['phi_bh'][()]
        ldot[i-dstart] = hfp['ldot'][()]
        fout[i-dstart] = hfp['fout'][()]
        hfp.close()
    
    print("\nAVERAGE FLUXES ({0:s} {1:3.2f}): ".format(model,spin))
    print("Accretion rate: {0:5.2f}".format(np.abs(mdot).mean()))
    print("Magnetic flux: {0:5.2f}".format(phi_bh.mean()/np.sqrt(np.abs(mdot).mean())))
    print("Angular momentum flux: {0:5.2f}".format(np.abs(ldot).mean()/np.abs(mdot).mean()))
    print("Energy flux: {0:5.4f}".format(np.abs(fout+mdot).mean()/np.abs(mdot).mean()))

    fig = plot.flux_series(t, mdot, phi_bh, ldot, fout+mdot)
    plt.savefig(os.path.join(model_dir,'fluxes_'+model+'_'+str(spin)+'.png'))
    plt.clf()

    # Disk-avg averaging
    print("\nDISC AVERAGED COMPUTATION ({0:s} {1:3.2f}) HAPPENING".format(model,spin))
    disk_avg_dir = os.path.join(model_dir,'disk_avg_stuff')
    variables = ['rho','pg','B','sigma','beta_inv']
    disk_dict = {}
    for var in variables:
        disk_dict[var] = np.zeros(r.size,dtype=float)
    for i in range(dstart,dend+1):
        hfp = h5py.File(os.path.join(disk_avg_dir,'disk_avg_radial_profiles_{0:04d}.h5'.format(i)),'r')
        disk_dict['rho'] += np.array(hfp['rho_avg'])
        disk_dict['pg'] += np.array(hfp['pg_avg'])
        disk_dict['B'] += np.array(hfp['B_avg'])
        disk_dict['sigma'] += np.array(hfp['sigma_avg'])
        disk_dict['beta_inv'] += np.array(hfp['beta_inv_avg'])
        hfp.close()
    
    for var in variables:
        disk_dict[var]/=Ndumps
    if os.path.isfile(os.path.join(data_dir,model,'disk_time_averaged_'+res+'.h5')):
        try:
            hfp = h5py.File(os.path.join(data_dir,model,'disk_time_averaged_'+res+'.h5'),'a')
            for var in variables:
                hfp[model+str(spin)+'/'+var] = disk_dict[var]
            hfp.close()
        except RuntimeError:
            print("Disc-averaged radial profiles for this model already written")
            pass
    else:
        hfp = h5py.File(os.path.join(data_dir,model,'disk_time_averaged_'+res+'.h5'),'w')
        for var in variables:
            hfp[model+str(spin)+'/'+var] = disk_dict[var]
        hfp.close()

    # Scale-height averaging
    print("\nSCALE HEIGHT CALCULATION ({0:s} {1:3.2f}) HAPPENING".format(model,spin))
    scale_dir = os.path.join(model_dir,'scale_height_stuff')
    variables = ['num','denom']
    scale_dict = {}
    for var in variables:
        scale_dict[var] = np.zeros(r.size,dtype=float)
    for i in range(dstart,dend+1):
        hfp = h5py.File(os.path.join(scale_dir,'scale_height_frac_values_{0:04d}.h5'.format(i)),'r')
        scale_dict['num'] += np.array(hfp['num'])
        scale_dict['denom'] += np.array(hfp['denom'])
        hfp.close()

    for var in variables:
        scale_dict[var]/=Ndumps
    if os.path.isfile(os.path.join(data_dir,model,'scale_height_time_averaged_'+res+'.h5')):
        try:
            hfp = h5py.File(os.path.join(data_dir,model,'scale_height_time_averaged_'+res+'.h5'),'a')
            for var in variables:
                hfp[model+str(spin)+'/'+var] = scale_dict[var]
            hfp.close()
        except RuntimeError:
            print("Scale height radial profiles for this model already written")
            pass
    else:
        hfp = h5py.File(os.path.join(data_dir,model,'scale_height_time_averaged_'+res+'.h5'),'w')
        for var in variables:
            hfp[model+str(spin)+'/'+var] = scale_dict[var]
        hfp.close()

    # Jet power computation
    print("\nJET POWER CALCULATION ({0:s} {1:3.2f}) HAPPENING".format(model,spin))
    if model == 'SANE':
        r_out = 40
    elif model == 'MAD':
        r_out = 100
    r_out_ind = np.argmax(grid['r'][:,0,0]-r_out>0)-1
    theta = grid['th'][r_out_ind,:,0]
    jp_dir = os.path.join(model_dir,'jet_power_stuff')
    rho = np.zeros(theta.size,dtype=float)
    bsq = np.zeros(theta.size,dtype=float)
    fdiff = np.zeros(theta.size,dtype=float)
    fout = np.zeros(theta.size,dtype=float)
    fm = np.zeros(theta.size,dtype=float)
    mdot = 0.0
    for i in range(dstart,dend+1):
        hfp = h5py.File(os.path.join(jp_dir,'jet_power_related_values_{0:04d}.h5'.format(i)),'r')
        rho += np.array(hfp['rho'])
        bsq += np.array(hfp['bsq'])
        fdiff += np.array(hfp['fdiff'])
        fout += np.array(hfp['fout'])
        fm += np.array(hfp['fm'])
        mdot += hfp['mdot'][()]
        hfp.close()

    rho/=Ndumps; bsq/=Ndumps; fdiff/=Ndumps; fout/=Ndumps; fm/=Ndumps; mdot/=Ndumps
    bg_sq = (fout/fm)**2-1; bg = np.sqrt(np.where(bg_sq<0,0,bg_sq))
    fig = plot.theta_profiles(rho, bsq, fdiff, bg, grid, r_out_ind)
    plt.savefig(os.path.join(model_dir,'theta_profiles_'+model+'_'+str(spin)+'.png'))
    plt.clf()

    bgcut = 1.
    fdiff = np.where((bg>bgcut) & ((theta<1) | (theta>np.pi-1)),fdiff,0)
    pjet = (np.sum(fdiff)*grid['dx2'])/mdot
    print("Jet power ({0:s} {1:3.2f}): {2:.2E}".format(model,spin,Decimal(pjet)))

    # Rotational profile calculation
    print("\nROTATIONAL PROFILE CALCULATION ({0:s} {1:3.2f}) HAPPENING".format(model,spin))
    omega_dir = os.path.join(model_dir,'rotational_stuff')
    num = np.zeros(r.size,dtype=float)
    denom = np.zeros(r.size,dtype=float)
    for i in range(dstart,dend+1):
        hfp = h5py.File(os.path.join(omega_dir,'omega_frac_values_{0:04d}.h5'.format(i)),'r')
        num += np.array(hfp['num'])
        denom += np.array(hfp['denom'])
        hfp.close()
    num/=Ndumps; denom/=Ndumps
    rmax = 100
    kep_fit = 1/(r**1.5+spin)
    fit = (1-(1-spin)**1/r**2)*(1/(r**1.5+spin))
    fig = plot.rotational_profile(num/denom, grid, rmax, kepfit=kep_fit, omegafit=fit)
    plt.savefig(os.path.join(model_dir,'rotational_profile_'+model+'_'+str(spin)+'.png'))
    plt.clf()
    



print("\n-------------------------------------------------------------")
print("\nPlotting disk-avg quantities")
rfile = h5py.File(os.path.join(data_dir,model,'radii_'+res+'.h5'),'r')
dfile = h5py.File(os.path.join(data_dir,model,'disk_time_averaged_'+res+'.h5'),'r')
variables = ['rho','pg','B','sigma','beta_inv']
models = []
radii_dict = {}
data_dict = {}
# ensuring spins in descending order
keys = list(rfile.keys()); keys_edited = []; model_len=len(model); 
for key in keys:
    keys_edited.append(float(key.encode("utf-8")[model_len:]))
keys = sorted(keys_edited)
for key in list(keys):
    models.append(key)
    radii_dict[str(key)] = rfile[model+str(key)][()]
    data_dict[str(key)] = {}
    for var in variables:
        data_dict[str(key)][var] = dfile[model+str(key)+'/'+var][()]
rfile.close()
dfile.close()
fig = plot.disk_avg_plot(radii_dict, data_dict, models, variables)
plt.savefig(os.path.join(data_dir,model,'disk_avg_profiles_'+res+'.png'))
plt.clf()

print("\nPlotting scale-height")
rfile = h5py.File(os.path.join(data_dir,model,'radii_'+res+'.h5'),'r')
sfile = h5py.File(os.path.join(data_dir,model,'scale_height_time_averaged_'+res+'.h5'),'r')
variables = ['num','denom']
models = []
radii_dict = {}
data_dict = {}
# ensuring spins in descending order
keys = list(rfile.keys()); keys_edited = []; model_len=len(model);
for key in keys:
    keys_edited.append(float(key.encode("utf-8")[model_len:]))
keys = sorted(keys_edited)
for key in list(keys):
    models.append(key)
    radii_dict[str(key)] = rfile[model+str(key)][()]
    data_dict[str(key)] = {}
    for var in variables:
        data_dict[str(key)][var] = sfile[model+str(key)+'/'+var][()]
rfile.close()
sfile.close()
fig = plot.scale_height(radii_dict, data_dict, models)
plt.savefig(os.path.join(data_dir,model,'scale_height_profiles_'+res+'.png'))
plt.clf()
