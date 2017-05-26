#Use some decorators or closures to deal with inputs that are custom for each run
#

def do_climate_thing(path, climate_arguments):
    '''
    Load climate data year by year, we need tasmax by IR. We have this data. 
    Per run, do this 100 times, read a new dataset in, and take the dot-product
    Load climate data from various inputs

    Parameters
    ----------
    path: str
        path to a netCDF file
        
    climate_arguments: str
        climate variable evaluated 
        
    Returns
    -------
    DataArray

    '''  
    ds = xr.open_dataset(path)
    return ds[climate_arguments]
    
    
    
def do_covariate_thing(*paths):
    '''
    Read in covariate data
    Align datasets along region dimension
    Take IR-level baselines
    Perform any necessary transformations
    For socio-economic data we need to reevaluate every few years 
    
    Parameters
    ----------
    Paths to covariate data
    
    Returns
    -------
    MxN Matrix where M is the number of IRs and N is the number of covariates
    
    '''
    
    #read file
    #take baselines
    #update parameters
    
        
def get_local_surface_estimate_at_given_p_value(IR_annual_covariates, gammas, pval):
    '''
    Element-wise multiplication of gammas and IR-level covariates. 
    Equivalent to the local response functions

    
    Parameters
    ----------
    IR_covariates: MxN matrix where M is the num IRs and N is the covariates
    
    gammas: Mx1 column vector of point estimates of global impact function specification parameters
    
    Returns
    -------
    MxN Matrix where M is IRs and N is local estimate of function specification params

    '''
    
    #Load IR matrix
    #load gammas
    #do math
    #return matrix maybe write to disk or do some diagnostic
    
    
    
    
def do_math_thing(climate_object, IR_parameters,function_specification):
    '''
    Do math according to function specification
    
    Parameters
    ----------
    Climate_object: Xarray DataSet or Xarray DataArray
    
    IR_parameters: MxN matrix of regions by parameters 
    
    function_specification: Arbitrary function that takes Climate_object and IR_params as args
    
    Returns
    -------
    Xarray object: result of computation as a MxN Matrix where M is the number of IRs and N is number responses
                   represents one year of impacts for each IR
    
    .. note:: Many of the above recipes can be optimized by replacing global lookups with
    local variables defined as default values. For example, the dotproduct recipe can be written as
    
        def dotproduct(vec1, vec2, sum=sum, imap=imap, mul=operator.mul):
            return sum(imap(mul, vec1, vec2))
    
    '''
    
    return function_specification(climate_object, IR_parameters)

def do_collect_results_thing(annual_responses_by_IR_object, temp_path_on_disk, output_dir=None):
    '''
    Collects annual results into larger dataset
    
    Parameters
    ----------
    annual_responses_by_IR_object: Xarray DataSet
        represents annual impacts
        
    temp_path_on_disk
    -----------------
        
    output_dir: str
        Location to write output
    
    Returns
    -------
    N-dimensional data set or None
    
    '''
    #append annual_responses_by_IR_object to file on disk
    ds = xr.open_dataset(path_on_disk)
    #ds.update(annual_response_by_IR_object) 
    
    if output_dir is not None:
        ds.write_to_netcdf(output_dir)
    return ds
    

def to_datafs(api_object, archive_name, ds= None,path=None, cache=False):
    '''
    Creates a record in datafs and uploads file to osdc. If cache is True will save to cache
    
    Parameters
    ----------
    api_object: initialized api with users parameters from local datafs config file
    
    archive_name: str
        name of archive to create in datafs
        
    ds: Xarray Object
        dataset to be created and uploaded to datafs 
    
    path: str
        if None then archive is created from memory
    
    cache=bool
        if True, save file to cache
    
    '''

def do_pipeline(init_dict):
    '''
    Parameters
    ----------
    init_dict: dict
        Specifies the parameters of the subroutines in do_pipeline  
        
    Returns
    -------
    None
    Creates archive(s) in datafs that correspond to parameters in init_dict
    
    
    1.get coefficients from the csvv draw via np.multinomial.rvs(gammas, covariates)
    2.multiply coefficients by values in socioeconomics for each region 
    3.when we multiply the IR-level socio values and the gammas we get an IR-level curve
    4.take the dot product of the climate variable at that order of magnitude 
    '''
    
    