from RARL_distributed_def_comp import distributed_RARL

lambda_list = [0.00, 0.25, 0.50, 0.75, 1.00]
#step_size_list = [0.0001, 0.0002, 0.0004, 0.0005, 0.001, 0.003, 0.005, 0.01] 
step_size_list = [0.0001]
lambda_list = [0.25]
for l in lambda_list:
    for s in step_size_list:
        distributed_RARL(Lambda=l, step_size=s)
