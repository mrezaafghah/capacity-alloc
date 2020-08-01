import numpy as np

# we generate three matrices for demands of different services
# for T time intervals

time = 360
# standard demand
mu_r_std = 50   # average of reserved demands
mu_o_std = 60  # average of on-demand
mu_s_std = 300  # average of spot demand
mu_u_std = 0.5  # average of utilization

# drought demand
mu_r_drt = 50   # average of reserved demands
mu_o_drt = 60  # average of on-demand
mu_s_drt = 300  # average of spot demand
mu_u_drt = 0.5  # average of utilization

std_u = 0.15  # standard deviation of utilization
flag = False

for j in range(5):
    # defining the matrix of demands
    d = np.zeros(shape=[4, time])

    for i in range(time):
        # holiday
        if (i+1) in range(187, 201):
            # adding vector of reservation demand
            d[0, i] = np.random.poisson(mu_r_drt)

            # adding vector of pay-as-you-go demand
            d[1, i] = np.random.poisson(mu_o_drt)

            # adding vector of spot instances demand
            d[2, i] = np.random.poisson(mu_s_drt)

            # adding the vector of utilization
            while flag is False:
                d[3, i] = np.random.normal(mu_u_drt, std_u)   # utilization of the reserved instances
                if 1 > d[3, i] > 0:
                    flag = True
            flag = False

        else:
            # weekends: (drought demand)
            if (i+1) % 7 in (6, 0):
                # drought time:
                # adding vector of reservation demand
                d[0, i] = np.random.poisson(mu_r_drt)

                # adding vector of pay-as-you-go demand
                d[1, i] = np.random.poisson(mu_o_drt)

                # adding vector of spot instances demand
                d[2, i] = np.random.poisson(mu_s_drt)

                # adding the vector of utilization
                while flag is False:
                    d[3, i] = np.random.normal(mu_u_drt, std_u)   # utilization of the reserved instances
                    if 1 > d[3, i] > 0:
                        flag = True
                flag = False

            else:
                # standard time:
                # adding vector of reservation demand
                d[0, i] = np.random.poisson(mu_r_std)

                # adding vector of pay-as-you-go demand
                d[1, i] = np.random.poisson(mu_o_std)

                # adding vector of spot instances demand
                d[2, i] = np.random.poisson(mu_s_std)

                # adding the vector of utilization
                while flag is False:
                    d[3, i] = np.random.normal(mu_u_std, std_u)   # utilization of the reserved instances
                    if 1 > d[3, i] > 0:
                        flag = True
                flag = False

    # storing the demands
    t = j+1
    filename = "data" + str(t)
    demand = open('%s.csv' % filename, 'w')
    for w in range(time):
        demand.write(" %s" % d[0, w])
        demand.write(", %s" % d[1, w])
        demand.write(", %s" % d[2, w])
        demand.write(", %s \n" % d[3, w])
    demand.close()
