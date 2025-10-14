import numpy as np
import pandas as pd

from typing import Callable, List, Tuple

class Function:
    def __init__(self, func: Callable, x0: float, x1: float, epsilon: float, maxit: int=20, derivative: Callable=None):
        """
        ARGS:
            func = Função analisada **(deve ser passada como função lambda)**
            x0 = Valor de x0 (menor)
            x1 = Valor de x1 (maior)
            epsilon = Tolerância de erro
            maxit = Máximo de iterações
            derivative (opcional) = Derivada da função **(essencial para o método de Newton)**
        """
        self.func = func
        self.x0 = x0
        self.x1 = x1
        self.epsilon = epsilon
        self.maxit = maxit
        self.derivative = derivative

    def _table(self, data: List[float], titles: List[str]) -> pd.DataFrame:
        """
        Cria a tabela com as informações de cada iteração

        ARGS:
            title = Título da tabela
            values = Valores da tabela
        """
        pd.options.display.float_format = '{:.5f}'.format
        return pd.DataFrame(data, columns=titles)

    def bisect(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula a aproximação da raiz da função pelo método da bisseção.
        """
        a, b = self.x0, self.x1
        it = self.maxit
        table_title = ['k', 'xk', 'f(xk)', 'step']
        table = []

        for i in range(it):
            step = abs(b - a)
            if (step < self.epsilon):
                break
            # Bisseção
            c = (a + b) / 2
            f_c = self.func(c)
            table.append([i + 1, c, f_c, step])
            # Lógica de redução
            if (self.func(a) * f_c < 0):
                b = c
            else:
                a = c
            
        full_table = self._table(table, table_title)
        return f_c, full_table

    def newton(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula a aproximação da raiz da função pelo método de Newton.
        """
        if (self.derivative is None):
            raise ValueError("A derivada da função é necessária para o método de Newton.")
        
        table_title = ('k', 'xk', 'f(xk)', 'f\'(a)', 'step')

    def secant(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula a aproximação da raiz da função pelo método da secante.
        """
        table_title = ('k', 'xk', 'f(xk)', 'step')
        pass
