import numpy as np
import matplotlib.pyplot as plt

velPrev = 80
posPrev = 0
dt = 0.1
SIM_TIME = 50

A = np.array([ [1, dt],
               [0, 1]])
H = np.array([0, 1])
Q = np.array([ [1, 0],
               [0, 3]])
R = 10
x = np.array([0, 20])
P = 5*np.eye(2)

def getVel():
    global velPrev, posPrev

    v = 10*np.random.randn()
    velPrev = 80 + v
    posPrev = posPrev + velPrev * dt
    return velPrev, posPrev

def DvKalman(z):
    global x, P
    xEst = A @ x
    PEst = A @ P @ A.T + Q

    S = H @ PEst @ H.T +R
    K = (PEst @ H.T) /S

    x = xEst + ( K * (z- (H @ xEst) ) )
    P = (np.eye(2) - (K @ H))@ PEst

    return x, K, P

def main():
    print(__file__ + " Start !")

    time = 0.0

    hZVelMeas = [0]
    hZPos = [0]
    hxEst = [0, 0]
    hTime = [0]

    while SIM_TIME >= time:
        time += dt

        zvel, zpos = getVel()
        xEst, Kgain, cov =DvKalman(zvel)

        hZVelMeas = np.vstack( (hZVelMeas, zvel) )
        hxEst = np.vstack( (hxEst, xEst) )
        hTime = np.vstack( (hTime, time ) )
        hZPos = np.vstack( (hZPos, zpos) )

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(hTime, hZVelMeas, 'r--')
    axs[0].plot(hTime, hxEst[:,1])
    axs[0].set_title('measurements and the estimated velocity')
    axs[1].plot(hTime, hZPos, 'r--')
    axs[1].plot(hTime, hxEst[:,0])
    axs[1].set_title(' estimated position by kalman filter')
    # plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()











