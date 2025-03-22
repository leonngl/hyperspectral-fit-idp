import numpy as np
import concurrent.futures
import scipy.linalg
import time


def test_func(arr):
    m = np.random.rand(2000, 100)
    m_pinv = scipy.linalg.pinv(m)
    res = m_pinv @ arr
    err = arr - m @ res
    return res, err


A = np.random.rand(2000, 7000)

c = np.empty((100, 700))
errors = np.empty((200, 700))

with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    idx = 0
    for res, err in executor.map(test_func, np.array_split(A, 6, axis=1)):
        cur_len = res.shape[1]
        c[:, idx:idx + cur_len]
        idx += cur_len
        print("Iteration done.")

print(A)
print("Done.")


