from Parameters import *
#from Models import *
from trident import LightRay

from time import time

t0 = time()

fields_ = fields[:]
fields_.remove('Density')
fields_.append('density')

def MakeFullRay( i, lr=None ):
    if lr is None:
        lr = LightRay(   # load LightRay model for each ray to use different seed
            param_file,  
            simulation_type="Enzo",
            near_redshift=0, ## in this redshift range  !!!!
            far_redshift=min( redshift_max, ds0.current_redshift ),
            find_outputs=True,     # find the snapshots
            #        max_box_fraction=max(box_fractions),
            max_box_fraction=0.5,
            minimum_coherent_box_fraction=0.1,
            use_minimum_datasets=False
        )
    filename = root_rays + model + '/ray_far%i_%i.h5' % (i,npix)
    lr.make_light_ray( # compute LoS and save to .h5
        seed= (i+1)*seed,
        fields=fields_[:],
        data_filename = filename,
        use_peculiar_velocity=False,  # do not correct redshift for doppler shift from peculiar velocity  
#        left_edge=border[0],
#        right_edge=border[1],
        periodic=periodic,
    )


lr = LightRay(   # load LightRay model for each ray to use different seed
    param_file,  
    simulation_type="Enzo",
    near_redshift=0, ## in this redshift range  !!!!
    far_redshift=min( redshift_max, ds0.current_redshift ),
    find_outputs=True,     # find the snapshots
#        max_box_fraction=max(box_fractions),
    max_box_fraction=0.5,
    minimum_coherent_box_fraction=0.1,
    use_minimum_datasets=False
)

MakeFullRay( 4, lr=lr )
#MakeFullRay( 5, lr=lr )

print 'took %.0f second' % ( time() - t0 )
