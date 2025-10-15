import numpy as np
import pandas as pd

from methods import Function

def main():
    f = lambda x: np.cos(x, dtype="float16") - np.sin(x, dtype="float16")
    fd = lambda x: -1 * (np.sin(x, dtype="float16") + np.cos(x, dtype="float16"))
    x0 = 0.0
    x1 = np.pi / 2
    eps = 0.01
    maxit = 10
    func = Function(f, x0, x1, eps, maxit=maxit, derivative=fd)
    app, df = func.bisect()
    func.printFormated(df)
    print(f"\nAproximacao = {app}")
    print(df["f(xk)"][df.last_valid_index()])


if __name__ == "__main__":
    main()