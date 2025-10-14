import math as m
import pandas as pd

from methods import Function

def main():
    f = lambda x: m.cos(x) - m.sin(x)
    x0 = 0.0
    x1 = m.pi / 2
    eps = 0.01
    maxit = 7
    func = Function(f, x0, x1, eps, maxit)
    app, df = func.bisect()
    print(f"Aproximacao = {app}")
    print(f"DATAFRAME:\n{df}")

if __name__ == "__main__":
    main()