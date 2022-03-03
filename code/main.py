import error_rate_simulation as sim
import error_rate_function as func
import csv
import numpy as np
from scipy.optimize import curve_fit

f = open('data.csv', 'r')
r = csv.reader(f)
data = np.array(list(r))

f = open('data_avg.csv', 'r')
r = csv.reader(f)
data_avg = np.array(list(r))
f.close()

input = []
truth = []

for d in data:
    if(d[0] == '2'):
        input.append([d[5], d[2], d[1]])
        truth.append(d[3])

input = np.array(input, dtype=np.float32) / 1000
truth = np.array(truth, dtype=np.float32)


# free parameter of drift diffusion
# 0 : response boundary
# 1 : mean drift rate
# 2 : inter-trial standard deviation of drift rate
# 3 : range of starting point variability
# 4 : range of non-decision time
# 5 : mean non-decision time
free_parameter_dd = [0.0590, 0.2378, 0.1067, 0.0321, 0.0943, 0.1]

# free parameter of moving target acquisition
# 0 : implicit aim point
# 1 : action precision
# 2 : maximum reliability constant
# 3 : drift rate
free_parameter_mta = [0.3, 0.088, 0.042, 319.2]

# user parameter from experiment data
# 0 : Period of repetition
# 1 : Cue viewing time
# 2 : Cue within zone time (Tzone)
experiment_condition_value = [2, 0.35, 0.05]

#param = [free_parameter_dd, free_parameter_mta, experiment_condition_value]

#p = 0.0590, 0.2378, 0.1067, 0.0321, 0.0943, 0.1, 0.3, 0.088, 0.042, 319.2
p = 0.0590, 0.2378, 0.1067, 0.0321, 0.0943, 0.1

# print(func.error_rate(input, 0.0590, 0.2378, 0.1067, 0.0321, 0.0943, 0.1))
# print(sim.error_rate([p, free_parameter_mta, experiment_condition_value]))

### Model Curve Fitting ### 
popt, pcov = curve_fit(f=func.error_rate, xdata=input, ydata=truth, p0=p, epsfcn=0.01)

result_mta = func.mta_error_rate([0.3, 0.088, 0.042, 319.2], input)
result_avg = func.error_rate(input, 0.0590, 0.2378, 0.1067, 0.0321, 0.0943, 0.1)
result = func.error_rate(input, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

print("\n------------------------------")
### RMSE from MTA model ###
RMSE_mta = np.sqrt(np.mean((truth-result_mta)**2))
print(f"Old RMSE : {RMSE_mta}")

# RMSE_avg = np.sqrt(np.mean((truth-result_avg)**2))
# print(f"Avg param RMSE : {RMSE_avg}")

### RMSE from New fitted Model ###
RMSE_new = np.sqrt(np.mean((truth-result)**2))
print(f"New RMSE : {RMSE_new}")

