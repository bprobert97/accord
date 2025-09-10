# pylint: disable=too-many-locals too-many-nested-blocks
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

import numpy as np

def dstt_pred_mu_p(p0: np.ndarray, stm: np.ndarray, dstt: np.ndarray,
                   r: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance using the DSTTs.

    Args:
    - p0: Initial covariance matrix.
    - stm: State Transition Matrix.
    - dstt: Derivative of State Transition Tensor.
    - r: Rotation matrix.
    - dim: Dimension of the state space.

    Returns a tuple of:
    - mf: Mean vector after propagation.
    - pf: Covariance matrix after propagation.
    """
    n, m = len(stm), np.size(stm[0])
    r = np.asmatrix(r)
    p0r = np.matmul(np.matmul(r, p0), r.T)

    # Mean value
    mf = np.zeros([n])
    for i in range(n):
        for i1 in range(dim):
            for i2 in range(dim):
                mf[i] += dstt[i, i1, i2] * p0r[i1, i2] / 2

    # Covariance matrix
    pf = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            pf[i, j] = -mf[i] * mf[j]
            # First-order
            for a in range(m):
                for b in range(m):
                    pf[i, j] = pf[i, j] + stm[i, a] * stm[j, b] * p0r[a, b]
            # Second-order
            for a in range(dim):
                for b in range(dim):
                    for alpha in range(dim):
                        for beta in range(dim):
                            pf[i, j] = pf[i, j] + dstt[i, a, b] * dstt[j, alpha, beta] \
                                * (p0r[a, b] * p0r[alpha, beta] + p0r[a, alpha] \
                                   * p0r[b, beta] + p0r[a, beta] * p0r[b, alpha]) / 4
    return mf, pf
