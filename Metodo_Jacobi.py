"""
           Autor:
   Juan Pablo Buitrago Rios
   juanybrisagames@gmail.com
   Version 2.0 : 10/03/2025 11:50pm

"""

import numpy as np  # Importar la librería numpy para manejar arreglos y operaciones matemáticas
import matplotlib.pyplot as plt  # Importar la librería matplotlib para graficar resultados

# Definir el sistema de ecuaciones lineales mediante la matriz de coeficientes A y el vector b

#EJERCICIO 1.
A = np.array([[10, -1, 2, 0, 0],
              [-1, 11, -1, 3, 0],
              [2, -1, 10, -1, 0],
              [0, -1, 3, 8, -2],
              [0, 0, 2, -2, 10]])  # Matriz de coeficientes del sistema

b = np.array([6, 25, -11, 15, -10])  # Vector de términos independientes


"""
"EJERCICIO 2.
# Definir el sistema de ecuaciones lineales
A = np.array([[8, 2, -1, 0, 0, 0],
              [3, 15, -2, 1, 0, 0],
              [0, -2, 12, 2, -1, 0],
              [0, 1, -1, 9, -2, 1],
              [0, 0, -2, 3, 14, 1],
              [0, 0, 0, 1, -2, 10]])  # Matriz de coeficientes

b = np.array([10, 24, -18, 16, -9, 22])  # Vector de términos independientes
"""

"""
#EJERCICIO 3.
# Definir el sistema de ecuaciones lineales
A = np.array([[12, -2, 1, 0, 0, 0, 0],
              [-3, 18, -4, 2, 0, 0, 0],
              [1, -2, 16, -1, 1, 0, 0],
              [0, 2, -1, 11, -3, 1, 0],
              [0, 0, -2, 4, 15, -2, 1],
              [0, 0, 0, 1, -3, 2, 13]])  # Matriz de coeficientes

b = np.array([20, 35, -5, 19, -12, 25])  # Vector de términos independientes
"""

# Calcular la solución exacta del sistema utilizando la función solve de numpy
# Esto servirá para comparar el error de la solución obtenida por el método de Jacobi
sol_exacta = np.linalg.solve(A, b)  

# Definir criterios de paro para el método iterativo
# Se establece una tolerancia para la diferencia entre iteraciones y un número máximo de iteraciones

tolerancia = 1e-6  # Valor de tolerancia para la convergencia
max_iter = 100  # Número máximo de iteraciones permitidas

# Implementación del método iterativo de Jacobi
def jacobi(A, b, tol, max_iter):
    n = len(A)  # Número de ecuaciones (o tamaño de la matriz)
    x = np.zeros(n)  # Vector inicial de soluciones (inicializado en ceros)
    
    # Listas para almacenar errores en cada iteración
    errores_abs = []  # Error absoluto
    errores_rel = []  # Error relativo
    errores_cuad = []  # Error cuadrático
    
    for k in range(max_iter):  # Bucle principal de iteraciones
        x_new = np.zeros(n)  # Nuevo vector de soluciones en cada iteración
        
        for i in range(n):  # Iterar sobre cada ecuación
            # Sumar todas las contribuciones excepto la de la diagonal principal
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            # Calcular el nuevo valor de la variable i según la fórmula de Jacobi
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Calcular errores en cada iteración
        error_abs = np.linalg.norm(x_new - sol_exacta, ord=1)  # Norma 1 (error absoluto)
        error_rel = np.linalg.norm(x_new - sol_exacta, ord=1) / np.linalg.norm(sol_exacta, ord=1)  # Error relativo
        error_cuad = np.linalg.norm(x_new - sol_exacta, ord=2)  # Norma 2 (error cuadrático)
        
        # Guardar los errores en las listas correspondientes
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        # Imprimir los errores de la iteración actual
        print(f"Iteración {k+1}: Error absoluto = {error_abs:.6f}, Error relativo = {error_rel:.6f}, Error cuadrático = {error_cuad:.6f}")
        
        # Criterio de convergencia: detenerse si la diferencia máxima entre elementos es menor que la tolerancia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new  # Actualizar el vector de solución con los valores nuevos
    
    return x, errores_abs, errores_rel, errores_cuad, k+1  # Devolver la solución aproximada y los errores

# Ejecutar el método de Jacobi con la matriz A y el vector b definidos
sol_aprox, errores_abs, errores_rel, errores_cuad, iteraciones = jacobi(A, b, tolerancia, max_iter)

# Graficar la evolución de los errores en función de las iteraciones
plt.figure(figsize=(8,6))  # Tamaño de la figura
plt.plot(range(1, iteraciones+1), errores_abs, label="Error absoluto", marker='o')  # Graficar error absoluto
plt.plot(range(1, iteraciones+1), errores_rel, label="Error relativo", marker='s')  # Graficar error relativo
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='d')  # Graficar error cuadrático

plt.xlabel("Iteraciones")  # Etiqueta del eje x
plt.ylabel("Error")  # Etiqueta del eje y
plt.yscale("log")  # Escala logarítmica para mejor visualización de errores
plt.title("Convergencia de los errores en el método de Jacobi")  # Título del gráfico
plt.legend()  # Mostrar leyenda
plt.grid()  # Agregar una cuadrícula al gráfico
plt.savefig("errores_jacobi.png")  # Guardar la figura en un archivo PNG
plt.show()  # Mostrar el gráfico

# Imprimir la solución aproximada obtenida y la solución exacta para comparación
print(f"Solución aproximada: {sol_aprox}")
print(f"Solución exacta: {sol_exacta}")