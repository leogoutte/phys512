import numpy as np
import camb
import time

# to get spectrum (model) from CAMB parameters 
def get_spectrum(H0=0,ombh2=0,omch2=0,tau=0,As=0,ns=0,params=np.asarray([0,0,0,0,0,0]),lmax=3000,take_params=False):
    """
    Returns the spectrum for a given set of parameters H0, baryon density, dark matter density,
    optical depth, As, ns
    """
    if take_params:
        # define params
        H0=params[0]
        ombh2=params[1]
        omch2=params[2]
        tau=params[3]
        As=params[4]
        ns=params[5]

    # extract spectrum from camb
    params=camb.CAMBparams()
    params.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    params.InitPower.set_params(As=As,ns=ns,r=0)
    params.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(params)
    powers=results.get_cmb_power_spectra(params,CMB_unit='muK') # <- this is where we get the data
    cmb=powers['total']
    tt=cmb[:,0]

    return tt[2:]

# to reproduce the curvature matrix from the previous problem
def ndiff2(fun,x,dx_ord=0):
    """
    Numerical derivative using both +/- dx and +/- 2*dx
    """
    # this is a fine choice for our purposes:
    # a) Newton's method will keep iterating until the solution is good and
    # b) our function and its derivatives are of order unity
    order = -3 + dx_ord
    dx = 10**(order)
    
    # compute the function at the points of interest
    yplus = fun(x + dx)
    yminus = fun(x - dx)
    yplus2 = fun(x + 2*dx)
    yminus2 = fun(x - 2*dx)
    
    # compute the numerical derivative
    fprime = (8 * yplus - yplus2 + yminus2 - 8 * yminus) / (12 * dx)
    
    return fprime
def params_grad(fun,params):
    """
    Returns the numerical derivative at points `params` in parameter space
    """
    H0, ombh2, omch2, tau, As, ns = params
    
    # derivative w.r.t H0 at p
    fun_H00 = lambda H00: fun(H00,ombh2,omch2,tau,As,ns)
    grad_H0 = ndiff2(fun_H00,H0,dx_ord=+1)
    
    # derivative w.r.t ombh2 at p
    fun_ombh22 = lambda ombh22: fun(H0,ombh22,omch2,tau,As,ns)
    grad_ombh2 = ndiff2(fun_ombh22,ombh2,dx_ord=-2)

    # derivative w.r.t omch2 at p
    fun_omch22 = lambda omch22: fun(H0,ombh2,omch22,tau,As,ns)
    grad_omch2 = ndiff2(fun_omch22,omch2,dx_ord=-1)

    # derivative w.r.t tau at p
    fun_tauu = lambda tauu: fun(H0,ombh2,omch2,tauu,As,ns)
    grad_tau = ndiff2(fun_tauu,tau,dx_ord=-2)

    # derivative w.r.t As at p 
    fun_Ass = lambda Ass: fun(H0,ombh2,omch2,tau,Ass,ns) # sorry for the profanity
    grad_As = ndiff2(fun_Ass,As,dx_ord=-9)

    # derivative w.r.t ns at p
    fun_nss = lambda nss: fun(H0,ombh2,omch2,tau,As,nss)
    grad_ns = ndiff2(fun_nss,ns,dx_ord=0)

    # transpose to make it match with calc_lorentz
    return np.array([grad_H0, grad_ombh2, grad_omch2, grad_tau, grad_As, grad_ns]).T 

# computes chi squared
def chisquared(d,pred,errs):
    """
    Computes chi-squared given some data d and pred
    In our case, errs is a constant and is always the same
    """
    chi2 = np.sum((pred-d)**2/errs**2)
    
    return chi2

# takes a random step in parameter space
def random_step(cov):
    """
    Random Gaussian step in each parameter
    """
    scale = 1.5 # manually adjusted scale factor
    
    cov = cov/scale
    
    step = np.random.multivariate_normal(np.zeros(cov.shape[0]),cov)
    
    return step

# performs a mcmc step  
def mcmc_step(d,params,chisq,cov,errs):
    """
    Single step in MCMC chain
    Params is a vector 
    Param_errs is the covariance in the initial parameters in the 
    """
    # set constrained parameters
    tau_prior = 0.0540
    sigma_tau_prior = 0.0074

    # compute a set of new trial parameters 
    new_params = params + random_step(cov)
    
    # they predict the following data
    new_pred = get_spectrum(params=new_params,take_params=True)[:len(d)]
    
    # this data has the following chi squared
    # include the constraint that tau = tau_prior +/- sigma_tau_prior
    tau_chisq = ((new_params[3] - tau_prior)/sigma_tau_prior)**2
    new_chisq = chisquared(d,new_pred,errs) + tau_chisq
    
    # if it improves the chi squared, always accept
    # if not, accept it with a probability exp(-1/2* ((x^2)new - (x^2)old))
    log_accept_prob = -1/2*(new_chisq - chisq)
    
    # make it a log to avoid computing exponentials
    if np.log(np.random.rand(1)) < log_accept_prob:
        return new_params, new_chisq
    else:
        return params, chisq

# main mcmc function
def mcmc_main(d,initial_params,cov,errs,nstep=20000):
    """
    Main function for the MCMC chain
    """
    # initialize chain
    n = initial_params.size
    chain_params = np.zeros((nstep,n),dtype=float)
    chain_params[0,:] = initial_params
    
    # compute initial chi squared
    pred = get_spectrum(params=initial_params,take_params=True)[:len(d)]
    initial_chisq = chisquared(d,pred,errs)
    chain_chisq = np.zeros(nstep,dtype=float)
    chain_chisq[0] = initial_chisq
            
    # take `nstep` number of steps
    for i in range(1,nstep):
        # get the old parameters
        params = chain_params[i-1,:]
        chisq = chain_chisq[i-1]
        # compute the putatively new ones
        params_, chisq_ = mcmc_step(d,params,chisq,cov,errs)
        # put them into the chain
        chain_params[i,:] = params_
        chain_chisq[i] = chisq_
        
    return chain_params, chain_chisq

# code to run on terminal in camb environment
if __name__ == '__main__':
    # import data
    planck = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell = planck[:,0]
    spec = planck[:,1]
    errs = (planck[:,2] + planck[:,3]) / 2

    # initial guess
    p0 = np.asarray([69,0.022,0.12,0.054,2.1e-9,0.95])

    # get model spectrum
    pred = get_spectrum(params=p0,take_params=True)
    pred = pred[:len(spec)]
    params_newton = np.loadtxt('planck_fit_params.txt')[:,0] # for curvature matrix of previous problem
    grad = params_grad(fun=get_spectrum,params=params_newton)
    grad = grad[:len(spec)]

    # covariance matrix
    Ninv = np.linalg.inv(np.diag(errs**2))
    lhs = grad.T@Ninv@grad
    cov = np.linalg.inv(lhs)

    # run the mcmc simulation
    nsteps = 10000 # takes O(1) second per step, so this should take ~3 hours
    t1 = time.time()
    chain_params, chain_chisq = mcmc_main(d=spec,initial_params=p0,cov=cov,errs=errs,nstep=nsteps)
    t2 = time.time()

    # save the data
    np.savetxt("planck_chain_tau.txt", chain_params)
    np.savetxt("planck_chisq_tau.txt", chain_chisq)

    # report time
    print("Time it took to do {} MCMC steps:{}".format(nsteps,t2-t1))

