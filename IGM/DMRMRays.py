from Parameters import *
from Convenience import *
from Rays import GetRayData

keys = {
    'near'    : '/'.join(['',model_tag, 'near',str(nside) ] ),
    'far'     : '/'.join(['',model_tag, 'far', str(nside) ] ),
    'chopped' : '/'.join(['',model_tag, 'chopped' ] ),
}

def CollectDMRMRay( DMRM, key_new, remove=True ):
    with h5.File( DMRMrays_file, 'a' ) as ff:         ## open target file
        with h5.File( filename, 'r' ) as fil:  ##     open ray file
            f = fil['grid']
            for key in f.keys():          ##     for each data field
                new_key = '/'.join( [ key_new, key ] )
                try:                      ##       delete if present
                    ff.__delitem__(new_key)
                except:
                    pass                  ##       write ray data to target file
                ff.create_dataset(new_key, data=f[key].value )
                        
    if remove:
        os.remove( filename )             ##       remove ray file



def MakeDMRMRay( ipix, collect=False ):
    ## returns DM & RM along a single LoS
    DMRM = np.zeros( [2, len(redshift_skymaps[1:])] )
    iz = 0
    for z0, z in zip( redshift_skymaps, redshift_skymaps[1:] ):
        data = GetChoppedRayData_partly( ipix, z0, z1 )
        DMRM[iz] = GetDMRM( data, [z0,z1] )
        iz += 1
    if collect:
        CollectDMRMRay
    

## def MakeDMRMRays():
