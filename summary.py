import numpy as np
import confidence as cn

results = np.loadtxt(open("output.csv", "rb"), delimiter=",", skiprows=1)

rho = range(5)  # [0.1, 0.3, 0.5, 0.7, 0.9]
cap = range(3)  # [1800, 2200, 2600]
model = range(5)    # [1.0, 2.0, 2.1, 3.0, 3.1]
run_size = 360

average = np.zeros(shape=[5, 3, 5, 17])   # here we store the average of the output with indices indicating [Rho, Capacity, Model Version]
low_bound = np.zeros(shape=[5, 3, 5, 17])   # here we store the lower bound of the output with indices indicating [Rho, Capacity, Model Version]
up_bound = np.zeros(shape=[5, 3, 5, 17])   # here we store the upper bound of the output with indices indicating [Rho, Capacity, Model Version]

for i in rho:
    for j in cap:
        for k in model:
            temp = results[:run_size]
            temp = np.asarray(temp)
            results = results[run_size:]

            # stats of reserved instances:
            [demand_r_mean, demand_r_low, demand_r_up] = cn.mean_confidence_interval(temp[:, 0])

            [decision_r_mean, decision_r_low, decision_r_up] = cn.mean_confidence_interval(temp[:, 3])

            [state_r_mean, state_r_low, state_r_up] = cn.mean_confidence_interval(temp[:, 6])

            [state_lr_mean, state_lr_low, state_lr_up] = cn.mean_confidence_interval(temp[:, 7])

            # stats of on-demand instances:
            [demand_o_mean, demand_o_low, demand_o_up] = cn.mean_confidence_interval(temp[:, 1])

            [decision_o_mean, decision_o_low, decision_o_up] = cn.mean_confidence_interval(temp[:, 4])

            [state_o_mean, state_o_low, state_o_up] = cn.mean_confidence_interval(temp[:, 8])

            # stats of spot instances:
            [demand_s_mean, demand_s_low, demand_s_up] = cn.mean_confidence_interval(temp[:, 2])

            [decision_s_mean, decision_s_low, decision_s_up] = cn.mean_confidence_interval(temp[:, 5])

            [state_s_mean, state_s_low, state_s_up] = cn.mean_confidence_interval(temp[:, 9])

            [kick_s_mean, kick_s_low, kick_s_up] = cn.mean_confidence_interval(temp[:, 10])

            [thresh_s_mean, thresh_s_low, thresh_s_up] = cn.mean_confidence_interval(temp[:, 11])

            # average left capacity
            [left_c_mean, left_c_low, left_c_up] = cn.mean_confidence_interval(temp[:, 12])

            # stats on revenue:
            [rev_r_mean, rev_r_low, rev_r_up] = cn.mean_confidence_interval(temp[:, 13])

            [rev_o_mean, rev_o_low, rev_o_up] = cn.mean_confidence_interval(temp[:, 14])

            [rev_s_mean, rev_s_low, rev_s_up] = cn.mean_confidence_interval(temp[:, 15])

            [rev_t_mean, rev_t_low, rev_t_up] = cn.mean_confidence_interval(temp[:, 16])

            average[i][j][k] = [demand_r_mean, decision_r_mean, state_r_mean, state_lr_mean, demand_o_mean, decision_o_mean, state_o_mean, demand_s_mean, decision_s_mean, state_s_mean, kick_s_mean, thresh_s_mean, left_c_mean, rev_r_mean, rev_o_mean, rev_s_mean, rev_t_mean]

            low_bound[i][j][k] = [demand_r_low, decision_r_low, state_r_low, state_lr_low, demand_o_low, decision_o_low, state_o_low, demand_s_low, decision_s_low, state_s_low, kick_s_low, thresh_s_low, left_c_low, rev_r_low, rev_o_low, rev_s_low, rev_t_low]

            up_bound[i][j][k] = [demand_r_up, decision_r_up, state_r_up, state_lr_up, demand_o_up, decision_o_up, state_o_up, demand_s_up, decision_s_up, state_s_up, kick_s_up, thresh_s_up, left_c_up, rev_r_up, rev_o_up, rev_s_up, rev_t_up]


# finding the best model for each policy:
for j in cap:
    for i in rho:


# storing the results:
filename = "average_summary"
average_file = open('%s.csv' % filename, 'w')
average_file.write("Demand-Reserved, Decision-Reserved, State-Reserved, State-Reserved-Live, Demand-On-Demand, State-On-Demand, Decision-On-Demand, Demand-Spot, Decision-Spot, State-Spot, Kicked-out-Spot, Threshold, Left-Capacity, Revenue-Reserved, Revenue-On-Demand, Revenue-Spot, Total-Revenue, Rho, Capacity, Model-Version")


average_file.close()
