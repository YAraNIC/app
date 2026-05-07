import matplotlib.pyplot as plt

# Datos de tu tabla WCSS
iterations = [1, 2, 3]
wcss = [3200000, 2100000, 2050000]

# Crear gráfica
plt.plot(iterations, wcss, marker='o')
plt.title('WCSS Reduction Across Iterations')
plt.xlabel('Iteration')
plt.ylabel('WCSS')
plt.grid()

# Guardar imagen
plt.savefig('static/wcss_graph.png')

plt.close()

print("Graph created successfully!")