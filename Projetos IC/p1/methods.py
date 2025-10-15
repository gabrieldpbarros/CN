import numpy as np
import pandas as pd

from typing import Callable, List, Tuple

class Function:
    def __init__(self, func: Callable, x0: np.float16, x1: np.float16, epsilon: np.float16, maxit: int=20, derivative: Callable=None):
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
        pd.options.display.float_format = '{:.5E}'.format
        return pd.DataFrame(data, columns=titles)
    
    @staticmethod
    def printFormated(df: pd.DataFrame) -> None:
        """
        Imprime a tabela segundo a formatação requisitada pelo exercício.

        ARGS:
            df = Dataframe gerado pelo método numérico
        """
        print(df.to_string(index=False, justify='center'))


    def bisect(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula a aproximação da raiz da função pelo método da bisseção.
        """
        # Variáveis iniciais
        a, b = self.x0, self.x1
        table_title = ['k', 'xk', 'f(xk)', 'step']
        table = []
        # Loop principal
        for i in range(self.maxit):
            step = abs(b - a)
            # Bisseção
            xk = (a + b) / 2
            f_xk = self.func(xk)
            table.append([i + 1, xk, f_xk, step])
            # Lógica de redução
            if (self.func(a) * f_xk < 0):
                b = xk
            else:
                a = xk

            if (step < self.epsilon):
                break
            
        full_table = self._table(table, table_title)
        return f_xk, full_table

    def newton(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula a aproximação da raiz da função pelo método de Newton.
        """
        # Verificação inicial, a fim de evitar erro de divisão por None
        if not callable(self.derivative):
            raise ValueError("A derivada da função (derivative) não foi fornecida ou não é uma função.")
        # Variáveis iniciais  
        xk = self.x0
        table_title = ['k', 'xk', 'f(xk)', 'f\'(xk)', 'step']
        table = []
        # Loop principal
        for i in range(self.maxit):
            _xk = xk
            f_xk = self.func(xk)
            f_dxk = self.derivative(xk)
            xk = xk - (f_xk / f_dxk) # x_{k+1}
            step = abs(xk - _xk)

            table.append([i + 1, xk, self.func(xk), self.derivative(xk), step])

        full_table = self._table(table, table_title)
        return self.func(xk), full_table

    def secant(self) -> Tuple[float, pd.DataFrame]:
        """
        Calcula a aproximação da raiz da função pelo método da secante.
        """
        # Variáveis iniciais
        xk0, xk1 = self.x0, self.x1
        table = []
        table_title = ['k', 'xk', 'f(xk)', 'step']
        # Loop principal
        for i in range(self.maxit):
            f_xk0 = self.func(xk0)
            f_xk1 = self.func(xk1)

            den = xk0 * f_xk1 - xk1 * f_xk0
            num = f_xk1 - f_xk0

            xk0 = xk1 # x_k se torna x_{k-1}
            xk1 = den / num # x_k se torna x_{k-1}
            step = (xk1 - xk0)

            table.append([i + 1, xk1, self.func(xk1), step])

        full_table = self._table(table, table_title)
        return self.func(xk1), full_table