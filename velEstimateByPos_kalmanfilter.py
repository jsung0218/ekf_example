import numpy as np
import matplotlib.pyplot as plt

velPrev = 80
posPrev = 0
dt = 0.1
SIM_TIME = 50

A = np.array([ [1, dt],
               [0, 1]])
H = np.array([1, 0])
Q = np.array([ [10, 0],
               [0, 10]])
R = 100
x = np.array([0, 20])
P = 5*np.eye(2)

def getPosVel():
    global velPrev, posPrev

    w = 10*np.random.randn()
    v = 10*np.random.randn()

    z = posPrev + velPrev*dt + v
    posPrev = z - v
    velPrev = 80 + w
    return z, velPrev

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

    hZ = [0]
    hZVel = [0]
    hxEst = [0, 0]
    hTime = [0]

    while SIM_TIME >= time:
        time += dt

        z, zvel = getPosVel()
        xEst, Kgain, cov =DvKalman(z)

        hZ = np.vstack( (hZ, z) )
        hxEst = np.vstack( (hxEst, xEst) )
        hTime = np.vstack( (hTime, time ) )
        hZVel = np.vstack( (hZVel, zvel) )

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(hTime, hZ, 'r--')
    axs[0].plot(hTime, hxEst[:,0])
    axs[0].set_title('measurements and the estimated ')
    axs[1].plot(hTime, hZVel, 'r--')
    axs[1].plot(hTime, hxEst[:,1])
    axs[1].set_title('velocity estimated by kalman filter')
    # plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()











