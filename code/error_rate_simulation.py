import numpy as np
import math
import random

def drift_diffusion(fp_dd, timeout):
    correct_RTs=[]
    error_RTs=[]
    timeout_RTs=[]

    a = fp_dd[0]      # response boundary
    nu = fp_dd[1]     # mean drift rate
    eta = fp_dd[2]    # inter-trial standard deviation of drift rate
    sz = fp_dd[3]     # range of z variability 
    st = fp_dd[4]     # range of non-decision time
    Ter = fp_dd[5]    # mean non-decision time

    z = a * 0.5                         # starting point : middle
    s = 0.1                             # within-trial standard deviation of drift rate

    step_size = 0.00005                 # step size

    N_trials = 1000                     # number of trials

    for i in range(N_trials):
        while True:
            nu_sample = random.gauss(nu, eta)
            Ter_sample = Ter + random.random()*st - st/2
            z_sample = random.random()*sz - sz/2
            evidence = z + z_sample

            if nu_sample > 0 and Ter_sample > 0 and evidence > 0 and evidence < a:
                break

        p = 0.5*(1+nu_sample*(step_size**0.5/s))

        delta = s*(step_size**0.5)
        step_no = 0

        while True:
            sampling = np.random.binomial(1,p,None)

            if sampling == 1:
                evidence = evidence + delta
            elif sampling == 0:
                evidence = evidence - delta

            # print([evidence, a,i,nu_sample])

            step_no += 1

            if evidence >= a:
                correct_RTs.append(step_no*step_size+Ter_sample)
                break
            if evidence <= 0:
                error_RTs.append(step_no*step_size+Ter_sample)
                break
            if step_no * step_size > timeout - Ter_sample:
                timeout_RTs.append(evidence)
                break

    return (correct_RTs, error_RTs, timeout_RTs)

def dd_error_rate(fp_dd, fp_mta, ecv):
    # timeout = viewing time + aimpoint * zone time
    timeout = ecv[1] + fp_mta[0] * ecv[2]
    correct_RTs, error_RTs, timeout_RTs = drift_diffusion(fp_dd, timeout)

    num_correct = len(correct_RTs)
    num_error = len(error_RTs)

    # case 1: correct if evidence > boundary / 2
    # for timeout_RT in timeout_RTs:
    #     if timeout_RT >= (fp_dd[0] / 2):
    #         num_correct += 1
    #     else:
    #         num_error += 1

    # case 2: 50% miss 50% correct if forced to select
    # num_correct += len(timeout_RTs) * 0.5
    # num_error += len(timeout_RTs) * (1 - 0.5)

    # case 3: 100% error
    num_error += len(timeout_RTs)

    dd_er = num_error / (num_correct + num_error)
    print("E_DD : {}".format(dd_er))
    return dd_er

def mta_error_rate(fp_mta, ecv):
    c1 = fp_mta[0]       # c_mu : implicit aim point
    c2 = fp_mta[1]       # c_sigma : action precision
    c3 = fp_mta[2]       # maximum reliability constant
    c4 = fp_mta[3]       # drift rate

    P = ecv[0]        # Period of repetition
    tc = ecv[1]       # Cue viewing time
    tzone = ecv[2]    # Cue within zone time

    sigma_t = c2 * P

    if tc == 0:
        sigma_R = sigma_t
    else:
        sigma_v = 1/(np.exp(c4 * tc) - 1) + c3
        sigma_R = (sigma_t * sigma_v) / (((sigma_t ** 2) + (sigma_v ** 2)) ** 0.5)
    mu_R = c1 * tzone

    mta_er = 1 - 0.5 * (math.erf((tzone - mu_R)/(sigma_R * (2 ** 0.5))) + math.erf(mu_R/(sigma_R * (2 ** 0.5))))
    print("E_MTA : {}".format(mta_er))
    return mta_er

def error_rate(param):
    fp_dd = param[0]
    fp_mta = param[1]
    ecv = param[2]

    return 1 - (1 - mta_error_rate(fp_mta, ecv)) * (1 - dd_error_rate(fp_dd, fp_mta, ecv))
