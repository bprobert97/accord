# pylint: disable=too-many-locals, too-many-statements

"""
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Adapted from:
ZHOUXINGYU-OASIS, ‘ZHOUXINGYU-OASIS/MDSTT: v1.0.0’. Zenodo, Jul. 29, 2024.
doi: 10.5281/zenodo.13123587.
"""
import math
import numpy as np

def crtbp_dynamics(y: np.ndarray, mu: float) -> np.ndarray:
    """
    The dynamics of the CRTBP (Circular Restricted Three-Body Problem) model.

    Args:
    - y: State vector [x, y, z, vx, vy, vz].
    - mu: Mass parameter of the two primary bodies.

    Returns:
    - dydt: Time derivative of the state vector.
    """
    r1 = math.sqrt((mu + y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    r2 = math.sqrt((1 - mu - y[0]) ** 2 + (y[1]) ** 2 + (y[2]) ** 2)
    m1 = 1 - mu
    m2 = mu
    dydt = np.array([
        y[3],
        y[4],
        y[5],
        y[0] + 2 * y[4] + m1 * (-mu - y[0]) / (r1 ** 3) + m2 * (1 - mu - y[0]) / (r2 ** 3),
        y[1] - 2 * y[3] - m1 * (y[1]) / (r1 ** 3) - m2 * y[1] / (r2 ** 3),
        -m1 * y[2] / (r1 ** 3) - m2 * y[2] / (r2 ** 3)
    ])
    return dydt

def crtbp_dstt_dynamics(t: float,  # pylint: disable=unused-argument
                        y: np.ndarray,
                        mu: float,
                        r: np.ndarray,
                        dim: int) -> np.ndarray:
    """
    The dynamics of the CRTBP model (with STM and DSTT).

    Args:
    - t: Unused time argument. For compatibility with solve_ivp.
    - y: Augmented state vector including STM and DSTT.
    - mu: Mass parameter of the two primary bodies.
    - R: Rotation matrix.
    - dim: Dimension of the state space.

    Returns:
    - dy: Time derivative of the augmented state vector.
    """
    x = y[:6]
    stm = y[6:42].reshape(6, 6)
    dstt = y[42:].reshape(6, dim, dim)

    dxdt = crtbp_dynamics(x, mu)

    # STM
    n1 = cal_1st_tensor(x, mu)
    d_stm = np.matmul(n1, stm).reshape(36)

    # DSTM
    dstm = np.zeros([6, dim])
    for i in range(6):
        for k1 in range(dim):
            for l1 in range(dim):
                dstm[i, k1] = dstm[i, k1] + stm[i, l1] * r[k1, l1]

    # DSTT
    n2 = cal_2nd_tensor(x, mu)
    dstt = np.zeros([6, dim, dim])
    for i in range(6):
        for a in range(dim):
            for b in range(dim):
                for alpha in range(6):
                    dstt[i, a, b] = dstt[i, a, b] + n1[i, alpha] * dstt[alpha, a, b]
                    for beta in range(6):
                        dstt[i, a, b] = dstt[i, a, b] \
                            + n2[i, alpha, beta] * dstm[alpha, a] * dstm[beta, b]

    dstt_flat = dstt.reshape(6 * (dim ** 2))
    dy = np.concatenate((dxdt, d_stm, dstt_flat))
    return dy

def cal_1st_tensor(x: np.ndarray, mu: float) -> np.ndarray:
    """
    The first-order tensor of the CRTBP dynamics.

    Args:
    - x: State vector [x, y, z, vx, vy, vz].
    - mu: Mass parameter of the two primary bodies.

    Returns:
    - A: Jacobian matrix of the CRTBP dynamics.
    """
    rx = x[0]
    ry = x[1]
    rz = x[2]
    daxdrx = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) \
        - mu / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) \
            + (3 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) \
                / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
            - (3 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) \
                / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) + 1

    daxdry = (3 * mu * ry * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)

    daxdrz = (3 * mu * rz * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * rz * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)

    daxdvx = 0
    daxdvy = 2
    daxdvz = 0
    daydrx = (3 * mu * ry * (2 * mu + 2 * rx - 2)) \
        / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (3 * ry * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))

    daydry = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu \
        / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * ry ** 2) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * ry ** 2 * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) + 1

    daydrz = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)

    daydvx = -2
    daydvy = 0
    daydvz = 0

    dazdrx = (3 * mu * rz * (2 * mu + 2 * rx - 2)) \
        / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (3 * rz * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2))

    dazdry = (3 * mu * ry * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)

    dazdrz = (mu - 1) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) - mu \
        / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (3 / 2) + (3 * mu * rz ** 2) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz ** 2 * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)

    dazdvx = 0
    dazdvy = 0
    dazdvz = 0

    # Jacobian matrix
    a = np.zeros([6, 6])
    a[:3, 3:] = np.eye(3)
    a[3:, :] = np.array([
        [daxdrx, daxdry, daxdrz, daxdvx, daxdvy, daxdvz],
        [daydrx, daydry, daydrz, daydvx, daydvy, daydvz],
        [dazdrx, dazdry, dazdrz, dazdvx, dazdvy, dazdvz],
    ])
    return a

def cal_2nd_tensor(x: np.ndarray, mu: float) -> np.ndarray:
    """
    The second-order tensor of the CRTBP dynamics.

    Args:
    - x: State vector [x, y, z, vx, vy, vz].
    - mu: Mass parameter of the two primary bodies.

    Returns:
    - A: Second-order tensor of the CRTBP dynamics.
    """
    rx = x[0]
    ry = x[1]
    rz = x[2]
    a = np.zeros([6, 6, 6])

    # elements of A
    daxdrxrx = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
          - (3 * (mu - 1) * (2 * mu + 2 * rx)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
                + (3 * mu * (2 * mu + 2 * rx - 2)) \
                    / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
                      - (15 * mu * (mu + rx - 1) * (2 * mu + 2 * rx - 2) ** 2) \
                        / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
                            + (15 * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx) ** 2) \
                                / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daxdrxry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) \
                / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
                    + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) \
                        / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daxdrxrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daxdrxvx = 0
    daxdrxvy = 0
    daxdrxvz = 0

    daxdryrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) \
                / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
            + (15 * ry * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) \
                / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daxdryry = (3 * mu * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        + (15 * ry ** 2 * (mu + rx) * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
        - (15 * mu * ry ** 2 * (mu + rx - 1)) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daxdryrz = (15 * ry * rz * (mu + rx) * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
        - (15 * mu * ry * rz * (mu + rx - 1)) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daxdryvx = 0
    daxdryvy = 0
    daxdryvz = 0

    daxdrzrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (15 * mu * rz * (mu + rx - 1) * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        + (15 * rz * (mu + rx) * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daxdrzry = (15 * ry * rz * (mu + rx) * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
        - (15 * mu * ry * rz * (mu + rx - 1)) \
        / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daxdrzrz = (3 * mu * (mu + rx - 1)) \
        / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * (mu + rx) * (mu - 1)) \
            / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        + (15 * rz ** 2 * (mu + rx) * (mu - 1)) \
            / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
        - (15 * mu * rz ** 2 * (mu + rx - 1)) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daxdrzvx = 0
    daxdrzvy = 0
    daxdrzvz = 0
    daxdvxrx = 0
    daxdvxry = 0
    daxdvxrz = 0
    daxdvxvx = 0
    daxdvxvy = 0
    daxdvxvz = 0
    daxdvyrx = 0
    daxdvyry = 0
    daxdvyrz = 0
    daxdvyvx = 0
    daxdvyvy = 0
    daxdvyvz = 0
    daxdvzrx = 0
    daxdvzry = 0
    daxdvzrz = 0
    daxdvzvx = 0
    daxdvzvy = 0
    daxdvzvz = 0

    daydrxrx = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (15 * mu * ry * (2 * mu + 2 * rx - 2) ** 2) \
            / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        + (15 * ry * (mu - 1) * (2 * mu + 2 * rx) ** 2) \
            / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daydrxry = (3 * mu * (2 * mu + 2 * rx - 2)) \
        / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (3 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daydrxrz = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) \
        / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daydrxvx = 0
    daydrxvy = 0
    daydrxvz = 0
    daydryrx = (3 * mu * (2 * mu + 2 * rx - 2)) \
        / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (3 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (15 * mu * ry ** 2 * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        + (15 * ry ** 2 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daydryry = (9 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (9 * ry * (mu - 1)) \
            / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (15 * mu * ry ** 3) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * ry ** 3 * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daydryrz = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry ** 2 * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * ry ** 2 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daydryvx = 0
    daydryvy = 0
    daydryvz = 0

    daydrzrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) \
        / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    daydrzry = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * rz * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry ** 2 * rz) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * ry ** 2 * rz * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daydrzrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry * rz ** 2) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * ry * rz ** 2 * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    daydrzvx = 0
    daydrzvy = 0
    daydrzvz = 0
    daydvxrx = 0
    daydvxry = 0
    daydvxrz = 0
    daydvxvx = 0
    daydvxvy = 0
    daydvxvz = 0
    daydvyrx = 0
    daydvyry = 0
    daydvyrz = 0
    daydvyvx = 0
    daydvyvy = 0
    daydvyvz = 0
    daydvzrx = 0
    daydvzry = 0
    daydvzrz = 0
    daydvzvx = 0
    daydvzvy = 0
    daydvzvz = 0

    dazdrxrx = (3 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * rz * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * rz * (2 * mu + 2 * rx - 2) ** 2) \
                / (4 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
            + (15 * rz * (mu - 1) * (2 * mu + 2 * rx) ** 2) \
                / (4 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    dazdrxry = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) \
        / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    dazdrxrz = (3 * mu * (2 * mu + 2 * rx - 2)) \
        / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (3 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    dazdrxvx = 0
    dazdrxvy = 0
    dazdrxvz = 0

    dazdryrx = (15 * ry * rz * (mu - 1) * (2 * mu + 2 * rx)) \
        / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
        - (15 * mu * ry * rz * (2 * mu + 2 * rx - 2)) \
            / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    dazdryry = (3 * mu * rz) \
        / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) - (3 * rz * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry ** 2 * rz) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * ry ** 2 * rz * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    dazdryrz = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
        - (3 * ry * (mu - 1)) / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 \
                                          + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * ry * rz ** 2 * (mu - 1)) \
                    / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    dazdryvx = 0
    dazdryvy = 0
    dazdryvz = 0

    dazdrzrx = (3 * mu * (2 * mu + 2 * rx - 2)) \
        / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
        - (3 * (mu - 1) * (2 * mu + 2 * rx)) \
            / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2)) \
            - (15 * mu * rz ** 2 * (2 * mu + 2 * rx - 2)) \
                / (2 * ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)) \
            + (15 * rz ** 2 * (mu - 1) * (2 * mu + 2 * rx)) \
                / (2 * ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2))

    dazdrzry = (3 * mu * ry) / ((mu + rx - 1) ** 2 + ry ** 2 \
                                + rz ** 2) ** (5 / 2) - (3 * ry * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * ry * rz ** 2) / ((mu + rx - 1) ** 2 + ry ** 2 \
            + rz ** 2) ** (7 / 2) + (15 * ry * rz ** 2 * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    dazdrzrz = (9 * mu * rz) / ((mu + rx - 1) ** 2 + ry ** 2 \
                                + rz ** 2) ** (5 / 2) - (9 * rz * (mu - 1)) \
        / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (5 / 2) \
            - (15 * mu * rz ** 3) \
            / ((mu + rx - 1) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2) \
                + (15 * rz ** 3 * (mu - 1)) \
                / ((mu + rx) ** 2 + ry ** 2 + rz ** 2) ** (7 / 2)

    dazdrzvx = 0
    dazdrzvy = 0
    dazdrzvz = 0
    dazdvxrx = 0
    dazdvxry = 0
    dazdvxrz = 0
    dazdvxvx = 0
    dazdvxvy = 0
    dazdvxvz = 0
    dazdvyrx = 0
    dazdvyry = 0
    dazdvyrz = 0
    dazdvyvx = 0
    dazdvyvy = 0
    dazdvyvz = 0
    dazdvzrx = 0
    dazdvzry = 0
    dazdvzrz = 0
    dazdvzvx = 0
    dazdvzvy = 0
    dazdvzvz = 0

    a[3] = np.array([
        [daxdrxrx, daxdrxry, daxdrxrz, daxdrxvx, daxdrxvy, daxdrxvz],
        [daxdryrx, daxdryry, daxdryrz, daxdryvx, daxdryvy, daxdryvz],
        [daxdrzrx, daxdrzry, daxdrzrz, daxdrzvx, daxdrzvy, daxdrzvz],
        [daxdvxrx, daxdvxry, daxdvxrz, daxdvxvx, daxdvxvy, daxdvxvz],
        [daxdvyrx, daxdvyry, daxdvyrz, daxdvyvx, daxdvyvy, daxdvyvz],
        [daxdvzrx, daxdvzry, daxdvzrz, daxdvzvx, daxdvzvy, daxdvzvz],
    ])

    a[4] = np.array([
        [daydrxrx, daydrxry, daydrxrz, daydrxvx, daydrxvy, daydrxvz],
        [daydryrx, daydryry, daydryrz, daydryvx, daydryvy, daydryvz],
        [daydrzrx, daydrzry, daydrzrz, daydrzvx, daydrzvy, daydrzvz],
        [daydvxrx, daydvxry, daydvxrz, daydvxvx, daydvxvy, daydvxvz],
        [daydvyrx, daydvyry, daydvyrz, daydvyvx, daydvyvy, daydvyvz],
        [daydvzrx, daydvzry, daydvzrz, daydvzvx, daydvzvy, daydvzvz],
    ])

    a[5] = np.array([
        [dazdrxrx, dazdrxry, dazdrxrz, dazdrxvx, dazdrxvy, dazdrxvz],
        [dazdryrx, dazdryry, dazdryrz, dazdryvx, dazdryvy, dazdryvz],
        [dazdrzrx, dazdrzry, dazdrzrz, dazdrzvx, dazdrzvy, dazdrzvz],
        [dazdvxrx, dazdvxry, dazdvxrz, dazdvxvx, dazdvxvy, dazdvxvz],
        [dazdvyrx, dazdvyry, dazdvyrz, dazdvyvx, dazdvyvy, dazdvyvz],
        [dazdvzrx, dazdvzry, dazdvzrz, dazdvzvx, dazdvzvy, dazdvzvz],
    ])
    return a
