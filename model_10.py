import numpy as np
import math
import csv


def revenue_cal(
        trial, intervals, month_hour, capacity, rho, price,
        phi, alpha, on_demand_life_ave
):

    """
    defining the parameters:
    """
    def psi_t(x): return (intervals - x) / month_hour  # reservation discount factor

    """
    generating the demands
    """
    # reading the demand matrix
    filename = "data" + str(trial)
    reader = csv.reader(open('%s.csv' % filename, "rt"), delimiter=",")
    temp = list(reader)
    data = np.array(temp).astype("float")

    # storing demands of on-demand
    demand_r = (np.array(data)[:, 0]).astype("int")

    # storing demands of on-demand
    demand_o = (np.array(data)[:, 1]).astype("int")

    # storing demands of on-demand
    demand_s = (np.array(data)[:, 2]).astype("int")

    # storing the utilization at each period
    utilization = np.array(data)[:, 3]

    """
    Building the MDP and calculating the revenue at each time window
    """
    dec_r = np.zeros(shape=intervals, dtype=int)  # the decision at each time window for reserved instances
    dec_o = np.zeros(shape=intervals, dtype=int)  # the decision at each time window for on-demand instances
    dec_s = np.zeros(shape=intervals, dtype=int)  # the decision at each time window for spot instances
    s_r = np.zeros(shape=intervals, dtype=int)  # number of reserved instances at the system
    s_rl = np.zeros(shape=intervals, dtype=int)  # number of live reserved instances at the system
    s_o = np.zeros(shape=intervals, dtype=int)  # number of live on-demand instances at the system
    s_s = np.zeros(shape=intervals, dtype=int)  # number of live spot instances at the system
    s_c = np.zeros(shape=intervals, dtype=int)  # empty capacity in the system
    rev_r = np.zeros(shape=intervals)  # the array of revenue at all time windows for reserved instances
    rev_o = np.zeros(shape=intervals)  # the array of revenue at all time windows for reserved instances
    rev_s = np.zeros(shape=intervals)  # the array of revenue at all time windows for reserved instances
    rev_t = np.zeros(shape=intervals)  # the array of revenue at all time windows for reserved instances
    kicked_out = np.zeros(shape=intervals)  # the array of kicked out instances at each period
    threshold_s = np.zeros(shape=intervals)  # the threshold of spot instances at each time interval
    l_rt = 0    # number of active reserved instances initially
    l_ot = 0    # number of active on-demand instances initially
    h_j = np.zeros(shape=intervals)  # number of leaving on-demand instances at each time
    kick_out = 0    # the number of spot instances that are kicked out at each period
    threshold = 0.3144

    for i in range(intervals):
        """
        finding the decisions
        """
        r_t = min(capacity - (l_rt + l_ot), math.floor(demand_r[i] * rho))
        o_t = int(min(capacity - (r_t + l_rt + l_ot), demand_o[i]))
        s_t = int(min(capacity - (math.floor((r_t + l_rt) * utilization[i]) + o_t + l_ot), demand_s[i]))

        dec_r[i] = r_t
        dec_o[i] = o_t
        dec_s[i] = s_t

        """
        calculating the revenue at each time window
        """
        if i < (intervals - month_hour):
            rev_r[i] = r_t * phi + alpha * price * math.floor((r_t + l_rt) * utilization[i])
        else:
            rev_r[i] = psi_t(i) * r_t * phi + alpha * price * math.floor((r_t + l_rt) * utilization[i])

        rev_o[i] = (price * (o_t + l_ot))
        rev_s[i] = s_t * threshold

        rev_t[i] = (rev_r[i] + rev_o[i] + rev_s[i])

        # updating l_rt, l_ot and l_st
        l_rt = l_rt + r_t

        # the life time of on-demand instances
        for w in range(o_t):
            temp1 = np.random.geometric(p=on_demand_life_ave)
            if (i + temp1 + 2) < intervals:
                h_j[i + temp1] += 1

        # updating live on-demand instances regarding the leaving time
        l_ot += o_t

        # the capacity left in the system is
        left_capacity = capacity - (math.floor(l_rt * utilization[i]) + l_ot + s_t)

        # state of the system
        s_r[i] = l_rt
        s_rl[i] = math.floor(l_rt * utilization[i])
        s_o[i] = l_ot
        s_s[i] = s_t
        s_c[i] = left_capacity

        # threshold of spot instances
        threshold_s[i] = threshold

        # number of spot instances that were kicked out:
        kicked_out[i] = kick_out

        # updating the live reserved, on-demands after they leave the system:
        l_ot = l_ot - h_j[i]
        if i >= month_hour:
            l_rt = l_rt - dec_r[i - month_hour]

    return demand_r, demand_o, demand_s, dec_r, dec_o, dec_s, s_r, s_rl, s_o, s_s, s_c, rev_r, rev_o, rev_s, rev_t, threshold_s
