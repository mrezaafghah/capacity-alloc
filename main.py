import model_10 as mod10
import model_20 as mod20
import model_21 as mod21
import model_30 as mod30
import model_31 as mod31
import model_40 as mod40
import model_41 as mod41
import model_42 as mod42
import model_43 as mod43
import confidence as cn
import time
import numpy as np

"""
defining the parameters:
"""
slots = 360  # number of intervals that is being studied
mon_slots = 30  # number of intervals in one month
p = 2.592  # price of on-demand
fee = 45.1  # up front fee for reservation
dis_rate = 0.5741  # reservation discount rate
on_demand_mean = 0.09  # average chance of leaving the system for on_demand instances
spot_mean = 0.18  # average chance of leaving the system for spot instances
trials = 5  # number of trials of the simulation

"""
defining the decision variables
"""
dec_var = [0.1, 0.3, 0.5, 0.7, 0.9]  # the proportion of the reserved instances that will be accepted
Cap = [1800, 2200, 2600]  # the capacity of the servers

time0 = time.time()

"""
defining the outputs
"""
reserved_d = np.zeros(shape=(slots, trials), dtype=int)  # the demands of reserved instances
on_demand_d = np.zeros(shape=(slots, trials), dtype=int)  # the demands of on-demand instances
spot_d = np.zeros(shape=(slots, trials), dtype=int)  # the demands of spot instances
decision_r = np.zeros(shape=(slots, trials), dtype=int)  # the decision on reserved instances
decision_o = np.zeros(shape=(slots, trials), dtype=int)  # the decision on on-demand instances
decision_s = np.zeros(shape=(slots, trials), dtype=int)  # the decision on spot instances
state_r = np.zeros(shape=(slots, trials), dtype=int)  # the number of reserved instances
state_rl = np.zeros(shape=(slots, trials), dtype=int)  # the number of live reserved instances
state_o = np.zeros(shape=(slots, trials), dtype=int)  # the number of live on-demand instances
state_s = np.zeros(shape=(slots, trials), dtype=int)  # the number of live spot instances
state_c = np.zeros(shape=(slots, trials), dtype=int)  # remaining capacity
thresh_s = np.zeros(shape=(slots, trials))   # the threshold of spot instances
revenue_r = np.zeros(shape=(slots, trials))  # the revenue of reserved instances
revenue_o = np.zeros(shape=(slots, trials))  # the revenue of on-demand instances
revenue_s = np.zeros(shape=(slots, trials))  # the revenue of spot instances
revenue_t = np.zeros(shape=(slots, trials))  # the total revenue

# initiating the output file:
filename = 'output'
results_file = open('%s.csv' % filename, 'w')
results_file.write(
    "Demand-Reserved, Demand-On-Demand, Demand-Spot, Decision-Reserved, Decision-On-Demand, Decision-Spot,"
    " State-Reserved, State-Reserved-Live, State-On-Demand, State-Spot, Threshold, Left-Capacity, Revenue-Reserved,"
    " Revenue-On-Demand, Revenue-Spot, Total-Revenue, Rho, Capacity, Model-Version"
)

filename2 = 'revenue_total'
revenue_file = open('%s.csv' % filename2, 'w')
revenue_file.write("Average, Lower Bound, Upper Bound, Rho, Capacity, Model-Version")

# main loop
for q in range(np.size(dec_var)):
    for w in range(np.size(Cap)):
        # Model 1.0
        for x in range(trials):
            test = x + 1
            results = mod10.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q],
                price=p, phi=fee, alpha=dis_rate, on_demand_life_ave=on_demand_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 1.0"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 2.0
        for x in range(trials):
            test = x + 1
            results = mod20.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 2.0"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 2.1
        for x in range(trials):
            test = x + 1
            results = mod21.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 2.1"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 3.0
        for x in range(trials):
            test = x + 1
            results = mod30.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 3.0"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 3.1
        for x in range(trials):
            test = x + 1
            results = mod31.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 3.1"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 4.0
        for x in range(trials):
            test = x + 1
            results = mod40.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 4.0"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 4.1
        for x in range(trials):
            test = x + 1
            results = mod41.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 4.1"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 4.2
        for x in range(trials):
            test = x + 1
            results = mod42.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 4.2"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)

        # Model 4.3
        for x in range(trials):
            test = x + 1
            results = mod43.revenue_cal(
                trial=test, intervals=slots, month_hour=mon_slots, capacity=Cap[w], rho=dec_var[q], price=p, phi=fee,
                alpha=dis_rate, on_demand_life_ave=on_demand_mean, spot_life_ave=spot_mean
            )

            reserved_d[:, x] = np.reshape(results[0], newshape=slots)
            on_demand_d[:, x] = np.reshape(results[1], newshape=slots)
            spot_d[:, x] = np.reshape(results[2], newshape=slots)
            decision_r[:, x] = np.reshape(results[3], newshape=slots)
            decision_o[:, x] = np.reshape(results[4], newshape=slots)
            decision_s[:, x] = np.reshape(results[5], newshape=slots)
            state_r[:, x] = np.reshape(results[6], newshape=slots)
            state_rl[:, x] = np.reshape(results[7], newshape=slots)
            state_o[:, x] = np.reshape(results[8], newshape=slots)
            state_s[:, x] = np.reshape(results[9], newshape=slots)
            state_c[:, x] = np.reshape(results[10], newshape=slots)
            revenue_r[:, x] = np.reshape(results[11], newshape=slots)
            revenue_o[:, x] = np.reshape(results[12], newshape=slots)
            revenue_s[:, x] = np.reshape(results[13], newshape=slots)
            revenue_t[:, x] = np.reshape(results[14], newshape=slots)
            thresh_s[:, x] = np.reshape(results[15], newshape=slots)

        reserved_demand = np.mean(reserved_d, axis=1)  # the average demands of reserved instances
        on_demand_demand = np.mean(on_demand_d, axis=1)  # the average demands of on-demand instances
        spot_demand = np.mean(spot_d, axis=1)  # the average demands of spot instances
        decision_reserved = np.mean(decision_r, axis=1)  # the average decision on reserved instances
        decision_on_demand = np.mean(decision_o, axis=1)  # the average decision on on-demand instances
        decision_spot = np.mean(decision_s, axis=1)  # the average decision on spot instances
        state_reserved = np.mean(state_r, axis=1)  # the average number of reserved instances
        state_reserved_l = np.mean(state_rl, axis=1)  # the average number of live reserved instances
        state_on_demand = np.mean(state_o, axis=1)  # the average number of live on-demand instances
        state_spot = np.mean(state_s, axis=1)  # the average number of live spot instances
        state_capacity = np.mean(state_c, axis=1)  # remaining capacity
        revenue_reserved = np.mean(revenue_r, axis=1)  # the average revenue of reserved instances
        revenue_on_demand = np.mean(revenue_o, axis=1)  # the average revenue of on-demand instances
        revenue_spot = np.mean(revenue_s, axis=1)  # the average revenue of spot instances
        revenue_total = np.mean(revenue_t, axis=1)  # the average total revenue
        thresh_spot = np.mean(thresh_s, axis=1)     # the average threshold

        # Storing the outputs
        mod = "Model 4.3"
        for time_window in range(slots):
            results_file.write("\n %s" % reserved_demand[time_window])
            results_file.write(", %s" % on_demand_demand[time_window])
            results_file.write(", %s" % spot_demand[time_window])
            results_file.write(", %s" % decision_reserved[time_window])
            results_file.write(", %s" % decision_on_demand[time_window])
            results_file.write(", %s" % decision_spot[time_window])
            results_file.write(", %s" % state_reserved[time_window])
            results_file.write(", %s" % state_reserved_l[time_window])
            results_file.write(", %s" % state_on_demand[time_window])
            results_file.write(", %s" % state_spot[time_window])
            results_file.write(", %s" % thresh_spot[time_window])
            results_file.write(", %s" % state_capacity[time_window])
            results_file.write(", %s" % revenue_reserved[time_window])
            results_file.write(", %s" % revenue_on_demand[time_window])
            results_file.write(", %s" % revenue_spot[time_window])
            results_file.write(", %s" % revenue_total[time_window])
            results_file.write(", %s" % dec_var[q])
            results_file.write(", %s" % Cap[w])
            results_file.write(", %s" % mod)

        # calculating the confidence interval and average of total revenues & storing them for comparison
        sum_rev_t = np.sum(revenue_t, axis=0)
        [ave, low_bound, up_bound] = cn.mean_confidence_interval(sum_rev_t)

        revenue_file.write("\n %s" % ave)
        revenue_file.write(", %s" % low_bound)
        revenue_file.write(", %s" % up_bound)
        revenue_file.write(", %s" % dec_var[q])
        revenue_file.write(", %s" % Cap[w])
        revenue_file.write(", %s" % mod)



results_file.close()
revenue_file.close()

# measuring the time
time1 = time.time()
time_it_code = time1 - time0  # timing the run time of the main code
print(time_it_code)
