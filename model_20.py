import numpy as np
import math
import csv
from operator import itemgetter


def revenue_cal(
        trial, intervals, month_hour, capacity, rho, price,
        phi, alpha, on_demand_life_ave, spot_life_ave
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
    threshold_s = np.zeros(shape=intervals)  # the threshold of spot instances at each time interval
    l_rt = 0    # number of active reserved instances initially
    l_ot = 0    # number of active on-demand instances initially
    l_qt = 0    # number of active on-demands that got into the system after staying in queue
    h_j = np.zeros(shape=intervals)  # number of leaving on-demand instances at each time
    g_j = np.zeros(shape=intervals)  # number of leaving on-demand instances at each time which were from queue
    live_spot = []  # indicates the live spot instances in the system
    spot_pool = []  # the pool of the spot instances before accepting them to the system

    # queueing system parameters:
    q_number = np.zeros(shape=7, dtype=int)    # the on-demand instances in queue

    for i in range(intervals):
        """
        finding the decisions
        """
        l_st = np.size(live_spot, 0)    # number of live spot instances at the beginning of the period
        r_t = min(capacity - (l_rt + l_ot + l_qt), math.floor(demand_r[i] * rho))
        o_t = int(min(capacity - (r_t + l_rt + l_ot + l_qt), demand_o[i]))
        q_potential = demand_o[i] - o_t

        # we check the queue to see if we can add more on_demand
        q_t = 0     # the number of on-demands that are accepted in period t
        empty_cap = (capacity - (r_t + l_rt + o_t + l_ot + l_qt))
        if empty_cap > 0 and sum(q_number) > 0:
            flag = True
            indice = 6
            while flag and indice > -1:
                if (q_number[indice] - empty_cap) > 0:
                    q_t += empty_cap
                    q_number[indice] -= empty_cap
                    flag = False
                else:
                    q_t += q_number[indice]
                    empty_cap -= q_number[indice]
                    q_number[indice] = 0
                    indice -= 1
        q_t = int(0)

        # updating the number of instances in the queue
        for b in range(6):
            index = b + 1
            q_number[-index] = q_number[-(index+1)]
        q_number[0] = int(q_potential*np.random.uniform(low=0.3, high=0.6))

        # calculating the number of spot instances that will pass the threshold:
        # the array of the bids of spot instances and updating the array of live spot instances
        spot_price = np.random.normal(loc=0.3144+0.024, scale=0.03, size=demand_s[i])
        spot_price[spot_price < 0.1] = 0.1  # substituting those prices less than 0.1 with 0.1
        for u in range(demand_s[i]):
            temp2 = np.random.geometric(p=spot_life_ave)  # the lifetime of spot are half the on-demand on average
            spot_pool.append(np.array([spot_price[u], temp2, spot_price[u]*temp2]))

        # sorting the spot pool decreasingly (based on the bid)
        spot_pool = sorted(spot_pool, key=itemgetter(0), reverse=True)

        if (capacity - ((math.floor((r_t + l_rt) * utilization[i])) + o_t + l_ot + q_t + l_qt + l_st)) > 0:
            s_t = int(min(capacity - (math.floor((r_t + l_rt) * utilization[i]) + o_t + l_ot + q_t + l_qt + l_st), demand_s[i]))
        else:
            s_t = 0

        # fill the available spot instances into the system
        spot_number = int(capacity - ((math.floor((r_t + l_rt) * utilization[i])) + o_t + l_ot + q_t + l_qt))
        if spot_number < np.size(spot_pool, axis=0):
            live_spot = spot_pool[:spot_number]
        else:
            live_spot = spot_pool

        # spot pool now will only have those instances in live spot
        spot_pool = live_spot

        # number of live spot instances in the system
        l_st = np.size(live_spot, axis=0)

        dec_r[i] = r_t
        dec_o[i] = o_t + q_t
        dec_s[i] = s_t

        """
        calculating the revenue at each time window
        """
        if i < (intervals - month_hour):
            rev_r[i] = r_t * phi + alpha * price * math.floor((r_t + l_rt) * utilization[i])
        else:
            rev_r[i] = psi_t(i) * r_t * phi + alpha * price * math.floor((r_t + l_rt) * utilization[i])

        rev_o[i] = (price * (o_t + l_ot)) + (0.8 * price * (q_t + l_qt))
        rev_s[i] = np.sum(live_spot, axis=0)[0]

        rev_t[i] = (rev_r[i] + rev_o[i] + rev_s[i])

        # updating l_rt, l_ot and l_st
        l_rt = l_rt + r_t

        # the life time of on-demand instances
        for w in range(o_t):
            temp1 = np.random.geometric(p=on_demand_life_ave)
            if (i + temp1 + 2) < intervals:
                h_j[i + temp1] += 1

        # the life time of on-demand instances from the queue
        for w in range(q_t):
            temp3 = np.random.geometric(p=on_demand_life_ave)
            if (i + temp3 + 2) < intervals:
                g_j[i + temp3] += 1

        # updating live on-demand instances regarding the leaving time
        l_ot += o_t
        l_qt += q_t

        # updating live spot instances regarding the leaving time
        for q in range(np.size(live_spot, 0)):
            live_spot[q][1] -= 1
            live_spot[q][2] = live_spot[q][0]*live_spot[q][1]
        live_spot = [z for z in live_spot if z[1] > 0]

        # the capacity left in the system is
        left_capacity = capacity - (math.floor(l_rt * utilization[i]) + l_ot + l_qt + l_st)

        # state of the system
        s_r[i] = l_rt
        s_rl[i] = math.floor(l_rt * utilization[i])
        s_o[i] = l_ot + l_qt
        s_s[i] = l_st
        s_c[i] = left_capacity

        # threshold of spot instances
        if live_spot:
            threshold = live_spot[-1][0]
        else:
            threshold = price/3
        threshold_s[i] = threshold

        # updating the live reserved, on-demands after they leave the system:
        l_ot = l_ot - h_j[i]
        l_qt = l_qt - g_j[i]
        if i >= month_hour:
            l_rt = l_rt - dec_r[i - month_hour]

    return demand_r, demand_o, demand_s, dec_r, dec_o, dec_s, s_r, s_rl, s_o, s_s, s_c, rev_r, rev_o, rev_s, rev_t, threshold_s
