import numpy as np
import matplotlib.pyplot as plt

VoltConst = 14.4
A = 1
H = 1
Q = 0
R = 4
xx = 14
covValue = 6
K = 0

SIM_TIME = 50.0
DT = 0.2


def getVolt():
    w = np.random.randn()*R
    z = VoltConst + w
    return z

def simpleKalman(z):
    global covValue, xx, A, Q, R, H
    xp = A * xx
    Pp = A*covValue*np.transpose(A) + Q
    S = H*Pp*np.transpose(H) +R
    K = Pp * np.transpose(H) /(S)
    # K = Pp*np.transpose(H)*np.linalg.inv(S)

    x = xp + K*(z-H*xp)
    covValue = Pp - K*H*Pp

    return x, K, covValue

def main():
    print(__file__ + " Start !!")

    time = 0.0

    xEst = xx
    xMeas = 0

    hZ = xMeas
    hEst = xEst
    hTime = time
    hK = K
    hCov = covValue

    while SIM_TIME >= time:
        time += DT

        z = getVolt()
        volt, Kvalue, cov = simpleKalman(z)

        hEst = np.vstack( (hEst, volt) )
        hZ = np.vstack( (hZ, z) )
        hTime = np.vstack( (hTime, time) )
        hK = np.vstack( (hK, Kvalue) )
        hCov = np.vstack( (hCov, cov))

    # plt.cla()
    fig, axs = plt.subplots(2,1)
    axs[0].plot(hTime, hZ,'r--')
    axs[0].plot(hTime, hEst)
    axs[0].set_title('measurements and the estimated for constant voltage with noise')
    axs[1].plot(hTime, hK, 'r--')
    axs[1].plot(hTime, hCov)
    axs[1].set_title('kalman gain and variance')
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()



