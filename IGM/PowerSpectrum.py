import matplotlib
matplotlib.use('Agg')


## modified version of yt cookbook  https://yt-project.org/doc/cookbook

## taylored to obtain results from primordial (big snapshots, only central cube of 1/2 sidelength of interest)

import numpy as np
import matplotlib.pyplot as plt
import yt

"""
Make a turbulent KE power spectrum.  Since we are stratified, we use
a rho**(1/3) scaling to the velocity to get something that would
look Kolmogorov (if the turbulence were fully developed).

Ultimately, we aim to compute:

                      1  ^      ^*
     E(k) = integral  -  V(k) . V(k) dS
                      2

             n                                               ^
where V = rho  U is the density-weighted velocity field, and V is the
FFT of V.

(Note: sometimes we normalize by 1/volume to get a spectral
energy density spectrum).


"""


def doit(model):

    # a FFT operates on uniformly gridded data.  We'll use the yt
    # covering grid for this.

    ds = yt.load("/work/stuf315/MHD-models/clues/2018/primordial/%s/%s" % (model,model) )


    #max_level = ds.index.max_level
    ### due to memory shortage, only use limited max_level < 5
    ### as we are only interested in outer scale, level 0 should suffice
    max_level = 1

    # ref = int(np.product(ds.ref_factors[0:max_level]))
    ### depreciated, ds.ref_factors doesn't exist
    ref = max( 1, max_level * ds.refine_by )
    #ref = 1

    ### instead on starting at [0,0,0], start at [1/4,1/4,1/4] relative sidelength
    low = ds.domain_right_edge / 4
    dims = ds.domain_dimensions/2*ref

    nx, ny, nz = dims

    nindex_rho = 1./3.

    Kk = np.zeros( (nx//2+1, ny//2+1, nz//2+1))

    for vel in [("gas", "velocity_x"), ("gas", "velocity_y"),
                ("gas", "velocity_z")]:

        Kk += 0.5*fft_comp(ds, ("gas", "density"), vel,
                           nindex_rho, max_level, low, dims)

    # wavenumbers
    L = (ds.domain_right_edge - ds.domain_left_edge).d

    kx = np.fft.rfftfreq(nx)*nx/L[0]
    ky = np.fft.rfftfreq(ny)*ny/L[1]
    kz = np.fft.rfftfreq(nz)*nz/L[2]

    # physical limits to the wavenumbers
    kmin = np.min(1.0/L)
    kmax = np.min(0.5*dims/L)

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    whichbin = np.digitize(k.flat, kbins)
    ncount = np.bincount(whichbin)

    E_spectrum = np.zeros(len(ncount)-1)

    for n in range(1,len(ncount)):
        E_spectrum[n-1] = np.sum(Kk.flat[whichbin==n])

    k = 0.5*(kbins[0:N-1] + kbins[1:N])
    E_spectrum = E_spectrum[1:N]

    index = np.argmax(E_spectrum)
    kmax = k[index]
    Emax = E_spectrum[index]

    np.savetxt( "/work/stuf315/PreFRBLE/results/spectrum_%s.txt" % model, np.array([E_spectrum,k]).T, header='Energy spectrum\tk')
    
    plt.loglog( k, E_spectrum, label='z = %.2f' % ds.current_redshift )
    plt.loglog( k, Emax*(k/kmax)**(-5./3.), ls=":", color="0.5")

    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)dk$")
    plt.legend()

    plt.savefig("/work/stuf315/PreFRBLE/results/spectrum_%s.png" % model)

def fft_comp(ds, irho, iu, nindex_rho, level, low, delta ):

    cube = ds.covering_grid(level, left_edge=low,
                            dims=delta,
                            fields=[irho, iu])

    rho = cube[irho].d
    u = cube[iu].d

    nx, ny, nz = rho.shape

    # do the FFTs -- note that since our data is real, there will be
    # too much information here.  fftn puts the positive freq terms in
    # the first half of the axes -- that's what we keep.  Our
    # normalization has an '8' to account for this clipping to one
    # octant.
    ru = np.fft.fftn(rho**nindex_rho * u)[0:nx//2+1,0:ny//2+1,0:nz//2+1]
    ru = 8.0*ru/(nx*ny*nz)

    return np.abs(ru)**2



#ds = yt.load("/work/stuf315/MHD-models/clues/2018/primordial/RD0006/RD0006")
#ds = yt.load("/hummel/enzo/gas_plus_dm_std/RD0011/RedshiftOutput0011")
for model in ['RD00%02d' % i for i in [0,1,2,6,8,15,18,21]]:
    doit(model)
