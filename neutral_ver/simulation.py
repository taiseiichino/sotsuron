from RARL_distributed_def import distributed_RARL
#from RARL import RARL


"""step_size_list = [0.01, 0.005, 0.003, 0.001,0.0005, 0.0003, 0.0001, 0.0002]""" 
step_size_list = [0.002]

for s in step_size_list:
    distributed_RARL(step_size=s)

"""
for sp in slip_p_list:
    for s in step_size_list:
        
        RARL(Lambda=l, step_size=s, slip_p=sp)"""