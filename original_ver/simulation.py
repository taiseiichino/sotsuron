from RARL_distributed_def import distributed_RARL

lambda_list = [0.00]
step_size_list = [0.001] 

for l in lambda_list:
    for s in step_size_list:
        distributed_RARL(Lambda=l, step_size=s)