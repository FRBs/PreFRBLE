import h5py as h5, matplotlib.pyplot as plt, numpy as np

zs = np.arange(0.1,6.1,0.1)

with h5.File( 'results/LoS_observables.h5' ) as f:
    for i in range(128):
        SM = f['primordial/chopped/%i/SM_overestimate' % i].value
        print SM
        plt.plot( zs, SM )
plt.show()
plt.close()
