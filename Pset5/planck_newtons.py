import numpy as np
import camb

# define numerical derivative function

def ndiff2(fun,x,dx_ord=0):
    """
    Numerical derivative using both +/- dx and +/- 2*dx
    """
    # this is a fine choice for our purposes:
    # a) Newton's method will keep iterating until the solution is good and
    # b) our function and its derivatives are of order unity
    order = 3 + dx_ord
    dx = 10**(-order)
    
    # compute the function at the points of interest
    yplus = fun(x + dx)
    yminus = fun(x - dx)
    yplus2 = fun(x + 2*dx)
    yminus2 = fun(x - 2*dx)
    
    # compute the numerical derivative
    fprime = (8 * yplus - yplus2 + yminus2 - 8 * yminus) / (12 * dx)
    
    return fprime

# define funcion to extract spectrum from camb

def get_spectrum(H0=0,ombh2=0,omch2=0,tau=0,As=0,ns=0,params=np.asarray([0,0,0,0,0,0]),lmax=3000,take_params=False):
    """
    Returns the spectrum for a given set of parameters H0, baryon density, dark matter density,
    optical depth, As, ns
    """
    if take_params:
        # define params
        H0=pars[0]
        ombh2=pars[1]
        omch2=pars[2]
        tau=pars[3]
        As=pars[4]
        ns=pars[5]

    # extract spectrum from camb
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK') # <- this is where we get the data
    cmb=powers['total']
    tt=cmb[:,0]

    return tt

# define gradient helper function

def params_grad(fun,params):
    """
    Returns the numerical derivative at points `params` in parameter space
    """
    H0, ombh2, omch2, tau, As, ns = params
    
    # derivative w.r.t H0 at p
    fun_H00 = lambda H00: fun(H00,ombh2,omch2,tau,As,ns)
    grad_H0 = ndiff2(fun_H00,H0)
    
    # derivative w.r.t ombh2 at p
    fun_ombh22 = lambda ombh22: fun(H0,ombh22,omch2,tau,As,ns)
    grad_ombh2 = ndiff2(fun_ombh22,ombh2)

    # derivative w.r.t omch2 at p
    fun_omch22 = lambda omch22: fun(H0,ombh2,omch22,tau,As,ns)
    grad_omch2 = ndiff2(fun_omch22,H0)

    # derivative w.r.t tau at p
    fun_tauu = lambda tauu: fun(H0,ombh2,omch2,tauu,As,ns)
    grad_tau = ndiff2(fun_tauu,tau)

    # derivative w.r.t As at p 
    fun_Ass = lambda Ass: fun(H0,ombh2,omch2,tau,Ass,ns) # sorry for the profanity
    grad_As = ndiff2(fun_Ass,As)

    # derivative w.r.t ns at p
    fun_nss = lambda nss: fun(H0,ombh2,omch2,tau,As,nss)
    grad_ns = ndiff2(fun_nss,ns)

    # transpose to make it match with calc_lorentz
    return np.array([grad_H0, grad_ombh2, grad_omch2, grad_tau, grad_As, grad_ns]).T 

# define newtons method iterator
def newtons_method(p0,d,num,print_params=False):
    """
    Numerical derivative version of `newtons_method`
    Runs Newton's method for initial parameter guess p0
    t and d are the time and data, resp.
    """
    # starting parameters is p0
    p = p0.copy()

    for i in range(num):
        # calculate derivatives and function
        pred = get_spectrum(p)
        grad = params_grad(get_spectrum,p)
        
        # delta is difference between data and prediction
        r = d - pred
        err = (r**2).sum()
        r = r.T
        
        lhs=grad.T@grad
        rhs=grad.T@r
        dp=np.linalg.inv(lhs)@(rhs)
        for jj in range(p.size):
            p[jj]=p[jj]+dp[jj]
                        
        if print_params:
            print("The parameters are:",p)
            print("The step is:",dp)
    
    return p, dp


# test gradient function
params = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
grad = params_grad(get_spectrum, params)
print(grad)

# planck = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
# ell = planck[:,0]
# spec = planck[:,1]

# newtons_method(p0,d,num,print_params=False)





