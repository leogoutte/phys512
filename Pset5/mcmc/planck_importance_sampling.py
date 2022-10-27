import numpy as np
import camb
import time




# code to run on terminal in camb environment
if __name__ == '__main__':
    # import data
    planck = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
    ell = planck[:,0]
    spec = planck[:,1]
    errs = (planck[:,2] + planck[:,3]) / 2

    # import chains
    planck_chain = np.loadtxt("planck_chain.txt")
    planck_chisq = np.loadtxt("planck_chisq.txt")

    # define the new parameter value
    tau_prior = 0.0540
    sigma_tau_prior = 0.0074

    # compute the new chi2 (effective weight)
    # i.e. just the difference in our old taus and the new constrained one
    tau_chain = planck_chain[:,3]
    planck_chisq_new =  ((tau_prior-tau_chain)/sigma_tau_prior)**2

    # with this new chisquared, compute the new phase-space density L'/L
    density = np.exp(-0.5 * (planck_chisq_new))

    # compute the new parameters as weighted sums
    density_all = np.tile(density,(planck_chain.shape[1],1)).T / np.sum(density)
    params_weighted = np.sum(density_all * planck_chain, axis=0)

    # save the data
    np.savetxt("planck_chisq_new.txt", planck_chisq_new)
    np.savetxt("planck_density.txt", density)
    np.savetxt("planck_params_weighted.txt", params_weighted)
    np.savetxt("planck_chain_weighted.txt", density_all * planck_chain)

