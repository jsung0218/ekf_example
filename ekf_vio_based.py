

import math
import numpy as np
import matplotlib.pyplot as plt


# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)])**2

#  Simulation parameter
# Qsim = np.diag([0.2, np.deg2rad(1.0)])**2
# Rsim = np.diag([1.0, np.deg2rad(10.0)])**2
# Qsim = np.diag([0.2, np.deg2rad(1.0)])**2
# Rsim = np.diag([1.0, np.deg2rad(10.0)])**2
Qsim = np.array([ [0.1, 0],
               [0, 0.1]])
Rsim = np.array([ [0.1, 0],
               [0, 0.1]])

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 2  # State size [x,y]
LM_SIZE = 2  # LM state size [x,y]
CONTROL_SIZE = 2 # Control size [delta_x, delta_y]

show_animation = True

def motion_model(x, u):
    F = np.eye(STATE_SIZE)
    B = np.eye(CONTROL_SIZE)

    x = (F @ x) + (B @ u)
    return x, F, B

def calc_input(t):
    delta_x = math.cos(2*3.14*t/SIM_TIME)
    delta_y = math.sin(2*3.14*t/SIM_TIME)
    u = np.array([[delta_x, delta_y]]).T
    return u

def calc_innovation(lm, xEst, PEst, z):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0,0]
    zangle = math.atan2(delta[1,0], delta[0,0])
    zp = np.array([[ math.sqrt(q), pi_2_pi(zangle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacobH(q, delta, xEst)
    S = H @ PEst @ H.T + Rsim

    return y, S, H

def jacobH(q, delta, x):
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0]]])

    G = G / q

    H = G

    return H

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def observation(xTrue, xd, u, LM):
    xTrue, F, B = motion_model(xTrue, u)

    z = np.zeros((0,3))

    for i in range(len(LM[:,0])):
        dx = LM[i,0] - xTrue[0,0]
        dy = LM[i,1] - xTrue[1,0]
        d = math.sqrt(dx**2+dy**2)
        # angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        angle = pi_2_pi(math.atan2(dy, dx))
        dn = d + np.random.randn() *Qsim[0,0]
        anglen = angle + np.random.randn() *Qsim[1,1]
        zi = np.array([dn,anglen,i])
        z = np.vstack((z,zi))

    ud = np.array([[
        u[0,0] + np.random.randn() * Rsim[0,0],
        u[1,0] + np.random.randn() * Rsim[1, 1]]]).T
    xd, F, B = motion_model(xd, ud)
    return xTrue, z, xd, ud

def ekf_slam(xEst, PEst, u, z):

    # Predict
    xEst[0:STATE_SIZE], F, B = motion_model(xEst, u)
    PEst = F @ PEst @ F.T + Qsim

    # correction
    lm = np.zeros((0,2))
    lm = np.array([0.0, 0.0])
    y, S, H = calc_innovation(lm, xEst, PEst, z[1,0:2])
    K = (PEst @ H.T) @ np.linalg.inv(S)

    xEst = xEst + (K @ y)
    PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    return xEst, PEst


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[0.0, 0.0],
                     [15.0, 10.0],
                     # [3.0, 15.0],
                     # [-5.0, 20.0]
                     ])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input(time)

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # # plot landmark
            # for i in range(calc_n_LM(xEst)):
            #     plt.plot(xEst[STATE_SIZE + i * 2],
            #              xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()