import numpy as np
import matplotlib.pyplot as plt
import time

def validate_PCA(states, train_data):
  from sklearn.decomposition import PCA
  pca = PCA()
  pca.fit(train_data)
  true_matrix = pca.components_.T
  true_ev = pca.explained_variance_
  
  output_matrix = states['transform_matrix']
  error = np.mean(np.abs(np.abs(true_matrix) - np.abs(output_matrix)) / np.abs(true_matrix))
  if error > 0.01:
    print('Matrix is wrong! Error=',error)
  else:
    print('Matrix is correct! Error=', error)

  output_ev = states['eigen_vals']
  error = np.mean(np.abs(true_ev - output_ev) / true_ev)
  if error > 0.01:
    print('Variance is wrong! Error=', error)
  else:
    print('Variance is correct! Error=', error)

def train_PCA(train_data):
  ##### Implement here!! #####
  # Note: do NOT use sklearn here!
  # Hint: np.linalg.eig() might be useful
  states = {
      'transform_matrix': np.identity(train_data.shape[-1]),
      'eigen_vals': np.ones(train_data.shape[-1])
  }
  S = np.cov(train_data, rowvar=False, ddof=0)

  eig_vals, eig_vec = np.linalg.eig(S)
  index = np.flip(eig_vals.argsort())

  ##### Implement here!! #####
  states['transform_matrix'] = eig_vec[:, index]
  states['eigen_vals'] = eig_vals[index]

  return states


def faces(states, image_shape):
  fig = plt.figure()
  l = 2
  b = 5

  num = 10
  eigen_vecs = states["transform_matrix"][:, :num].T

  for i, n in enumerate(eigen_vecs):
    fig.add_subplot(l, b, i + 1)
    plt.imshow(n.reshape(*image_shape))
  plt.savefig("Eigen_faces_Q1_c")

def eigenface(states, effi):
  eig_val = states["eigen_vals"]
  length = len(eig_val)
  total = np.sum(eig_val)
  sum = 0
  count = 0
  for i in range(length):
    sum = sum + eig_val[i]
    effi_new = sum/total
    if(effi_new>effi):
      return count,effi_new
    count = count + 1



# Load data
start = time.time()
images = np.load('pca_data.npy')
num_data = images.shape[0]
train_data = images.reshape(num_data, -1)

states = train_PCA(train_data)
print('training time = %.1f sec'%(time.time() - start))

validate_PCA(states, train_data)

faces(states=states, image_shape=images[0].shape)


n_comp, efficiency= eigenface(states = states, effi = 0.95)
print("number of components for 95%", n_comp)
print("efficiency for 95%", efficiency)


n_comp_1, efficiency_1= eigenface(states = states, effi = 0.99)
print("number of components for 99%", n_comp_1)
print("efficiency for 99%", efficiency_1)


