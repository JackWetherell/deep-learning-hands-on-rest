import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iDEA


def random_potential(x, L, N=3, T=0.0795, D=1e-11, seed=None):
    '''
    Function to return random potential based on a set of Fourier terms and a confining term.

    Parameters:
    x (np.array): x grid points
    L (float):    Max width of system
    N (int):      Number of cos and sin terms (default = 3)
    T (float):    Damping factor of Fourier term (default = 0.0795)
    D (float):    Damping factor of confining terms (default = 1e-11)
    seed (int):   Seed for random numbers, if None no seed will be chosen (default=None)

    Returns:
    np.array: Randomly generated potential on given x grid.
    '''

    # Create seeds.
    if seed == None:
        s1 = None
        s2 = None
    else:
        s1 = seed
        s2 = seed + 1234 # Make sure the cos and sin terms are different.

    # Begin with confining term.
    V = np.power(x, 10) * D

    # Randomly generate coefficients.
    np.random.seed(s1)
    a = (np.random.random(N) - 0.5) * 2.0 * L / 3.0
    np.random.seed(s2)
    b = (np.random.random(N) - 0.5) * 2.0 * L / 3.0

    # Add Fourier terms.
    for i in range(N):
        V += T * a[i] * np.cos((float(i) * np.pi * x) / L)
        V += T * b[i] * np.sin((float(i) * np.pi * x) / L)

    return V


def generate(grid_points, count, seed=1):
    '''
    Generate a set of 2e data for the use of ML training.

    Parameters:
    grid_points (int):  Number of x grid points
    count (int):        Max width of system
    seed (int):         Seed of the random number generation (default = 1)

    Returns:
    None
    '''

    # Plot the first 5 potentials (on large grid).
    x = np.linspace(-15.0, 15.0, 1000)
    for i in range(0, 5):
        V = random_potential(x, L=15.0, seed=seed+i)
        plt.plot(x, V)
    plt.savefig('some_potentials.pdf')
    plt.gcf().clear()

    # Set up system.
    x = np.linspace(-15, 15, grid_points)
    v_ext = lambda x: 0.0*x # Placeholder.
    v_int = iDEA.interactions.softened_interaction
    s = iDEA.system.System(x, v_ext(x), v_int(x), NE=1, spin='u')

    # Check the parameters look reasonable.
    s.check() 

    # Initialise the lists.
    data_V = []
    data_E = []
    data_density = []

    # Generating the dataset.
    print('Generating {} entries:'.format(count))
    print('Progress:', end="", flush=True)
    for i in range(0, count):
        print('{0},'.format(i+1), end="", flush=True)

        # Generate the random potential.
        V = random_potential(s.x, L=15.0, seed=seed+i)

        # assign the random potential to the system
        s.v_ext = V

        # Solve the 1 electron system.
        orbitals, energies = iDEA.non_interacting.solve_states(s)
        density = iDEA.non_interacting.charge_density(s, orbitals)
        E = energies[0]

        # Add the data.
        data_V.append(V)
        data_E.append(E)
        data_density.append(density)

    # Convert to np array.
    data_V = np.array(data_V)
    data_E = np.array(data_E)
    data_density = np.array(data_density)

    # Save dataset.
    pickle.dump(data_V, open('V.db', 'wb'))
    pickle.dump(data_E, open('E.db', 'wb'))
    pickle.dump(data_density, open('density.db', 'wb'))
    print()

    # Plot some Densities.
    f, sub = plt.subplots(5,5)
    k = 0
    for i in range(5):
        for j in range(5):
            sub[i,j].plot(s.x, data_V[k,:], label='$V(x)$', color='red', linestyle='solid', linewidth=1.0)
            sub[i,j].plot(s.x, data_density[k,:], label='$n(x)$', color='black', linestyle='solid', linewidth=1.0)
            sub[i,j].axhline(y=data_E[k], color='b')
            sub[i,j].axis(ymin=-1.0,ymax=1.0)
            sub[i,j].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
            sub[i,j].set_xticklabels([])
            sub[i,j].set_yticklabels([])
            sub[i,j].set_xticks([], [])
            sub[i,j].set_yticks([], [])
            k = k + 1
    plt.savefig('data.pdf')
    plt.gcf().clear()


if __name__ == '__main__':
    generate(64, count=100000, seed=1)
