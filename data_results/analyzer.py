import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

def split_line(line):
  return [float(x) for x in line.split("\t")]

# Ler arquivo
with open("data.txt", 'r') as f:
    s = f.read().splitlines()

# Ler headers da primeira linha
headers = s[0].split('\t')
# Ler restante dos dados em matriz
data = np.array([split_line(line) for line in s[1:]])

# Alocar dados em vetores
# # Variáveis independentes
min_char_interval = data[:,0]
max_char_interval = data[:,1]
min_plate_interval = data[:,2]
max_plate_interval = data[:,3]

# # Variáveis dependentes
tempo = data[:,4]
acertos = data[:,10]

def plot1():

  with open("data_plate_thing.txt", 'r') as f:
    s = f.read().splitlines()

  non_zero_data = np.array([split_line(line) for line in s[1:] if split_line(line)[8] != 0])
  min_char_interval = non_zero_data[:,0]
  max_char_interval = non_zero_data[:,1]
  min_plate_interval = non_zero_data[:,2]
  max_plate_interval = non_zero_data[:,3]
  tempo = non_zero_data[:,4]
  acertos = non_zero_data[:,8]

  # Definição dos objetos de triangulação para plotagem
  triang = tri.Triangulation(min_char_interval, max_char_interval)

  # Plot 1 - Curvas de nível
  plt.subplot(1, 2, 1)
  plt.tricontourf(triang, acertos, levels=np.linspace(np.min(acertos), np.max(acertos),30))
  plt.colorbar()
  plt.tricontour(triang, acertos, colors='k')

def plot2():

  with open("data_plate_thing.txt", 'r') as f:
    s = f.read().splitlines()

  non_zero_data = np.array([split_line(line) for line in s[1:] if split_line(line)[8] != 0])
  min_char_interval = non_zero_data[:,0]
  max_char_interval = non_zero_data[:,1]
  min_plate_interval = non_zero_data[:,2]
  max_plate_interval = non_zero_data[:,3]
  tempo = non_zero_data[:,4]
  acertos = non_zero_data[:,8]

  # Definição dos objetos de triangulação para plotagem
  triang = tri.Triangulation(min_plate_interval, max_plate_interval)

  # Plot 1 - Curvas de nível
  plt.subplot(1, 2, 2)
  # plt.gca().set_aspect('equal')
  plt.tricontourf(triang, acertos, levels=np.linspace(np.min(acertos), np.max(acertos),30))
  plt.colorbar()
  plt.tricontour(triang, acertos, colors='k')
  # plt.title('Triangulação Delaunay para intervalos da placa x acertos')

def plot3():

  non_zero_data = np.array([split_line(line) for line in s[1:] if split_line(line)[10] != 0])
  min_char_interval = non_zero_data[:,0]
  max_char_interval = non_zero_data[:,1]
  min_plate_interval = non_zero_data[:,2]
  max_plate_interval = non_zero_data[:,3]
  tempo = non_zero_data[:,4]
  acertos = non_zero_data[:,10]

  # Plot 2 - Scatter bidimensional
  fig2,ax2 = plt.subplots()
  area = 300**(0.3+acertos)
  sc = ax2.scatter(min_char_interval, min_plate_interval, s=(tempo/50)**2, c=acertos, alpha=0.5, cmap=plt.cm.viridis_r)
  ax2.set_xlabel('Valor mínimo placas')
  ax2.set_ylabel('Valor mínimo caracteres')
  plt.colorbar(sc)

def others():
  # Redefinição das variáveis para plotagem
  x = x0
  y = y0
  colors = t
  area = 300**(0.3+z)
  # ax1.set_aspect('equal')

  # x = np.log10(t)
  # y = r
  # colors = x0
  # area = 300**(1-y0)  # 0 to 15 point radii

  # Redefinição das variáveis para plotagem
  x = x0
  y = y0
  z = r
  vx = xf/20
  vy = yf/20
  vz = 0
  colors = np.log10(t)
  colors -= np.min(colors)
  colors /= np.max(colors)
  colors = np.array((colors, 1 - colors, np.zeros(len(t)))).T
  area = 300**(0.3+z)  # 0 to 15 point radii

  # Plot 3 - Quiver tridimensional
  fig=plt.figure()
  ax3=fig.add_subplot(111, projection='3d')
  ax3.quiver(x,y,z,vx,vy,vz,color=colors)

  ax3.set_xlabel('X Label')
  ax3.set_ylabel('Y Label')
  ax3.set_zlabel('Z Label')
  ax3.set_title('Title')

fig, ax = plt.subplots(nrows=1, ncols=2)
plot1()
plot2()
plt.tight_layout()

plot3()
plt.show()