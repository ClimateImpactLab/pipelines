import xarray as xr
import pandas as pd




def labor_transform(ds, var):
    
    
    #rescale the data to kelvins
    ds[var] = ds[var] - 273.15
    
    for i in range(2,5):
        ds[var + str(i)] = ds[var]**i
        
    return ds
    
    
    