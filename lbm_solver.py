#!/usr/bin/python
""" Module LBM 2D
    Written by Ben Farcy
    Theory and doc references from:
    Sukop, Thorne, Lattice Boltzmann modeling, Springer, 2007
    Kruger et al., The Lattice Boltzmann Method Principles and Practice, Springer
"""

import json
import logging
import h5py
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

class Inputfile(object):
    """ inputfile infos """

    def __init__(self, filename):
        self.filename = filename
        self.read_inputfile()

    def read_inputfile(self):
        """ read inputfile """

        in_file = open(self.filename, 'r')
        self.dico = json.load(in_file)
        in_file.close()

        # Postproc 
        self.postproc_dump_niter = 50
        self.postproc_info_niter = 10
        self.postproc_vel_mag = "True"
        self.postproc_density = "True"
        self.postproc_vel_ux = "True"
        # Default dico
        #self.dico = dict(max_iter = 10000,
        #                 mesh = "mesh.dat",
        #                 reynolds = 105,
        #                 lu_x = 1.E-4,
        #                 scale = 2,
        #                 characteristic_dimension = 2.2E-3,
        #                 viscosity = 1.0E-6,
        #                 bnd_bottom = "wall_slip",
        #                 bnd_up = "wall_slip",
        #                 bnd_left = "inlet_neq",
        #                 bnd_right = "outlet",
        #                 tau = 0.6,
        #                 postproc_dump_niter = 50,
        #                 postproc_info_niter = 10,
        #                 postproc_vel_mag = True,
        #                 postproc_density = True,
        #                 postproc_vel_ux = True)

    def apply_inputfile(self, domain):
        """ apply inputfile """
        domain.reynolds = self.dico['reynolds']                   
        domain.lu_x = self.dico['lu_x'] 
        domain.scale = self.dico['scale']
        domain.characteristic_dimension = self.dico['characteristic_dimension'] 
        domain.viscosity = self.dico['viscosity']
        domain.bnd_bottom = self.dico['bnd_bottom']
        domain.bnd_up = self.dico['bnd_up']
        domain.bnd_left = self.dico['bnd_left'] 
        domain.bnd_right = self.dico['bnd_right']
        domain.tau = self.dico['tau']

class Domain(object):
    """ Domain definition """

    def __init__(self):
        self.solid = None  # boolean, dimensions (npx, npy)
        self.velocity = None  # dimensions: (2, npx, npy)
        self.init_velocity = None
        self.density = None  # dimensions: (npx, npy)
        self.characteristic_dimension_lu = None
        self.characteristic_dimension = None
        self.tau = None

        self.feq = None # Distribution function at equilibrium
        self.f_n = None # Distribution function at it=n
        self.f_tmp = None # Distribution function tmp


    def read_mesh(self, filename="mesh.dat"):
        """ Read the mesh """
        mesh_file = open(filename, 'r')
        mesh_lines = mesh_file.readlines() 

        np_mesh = numpy.array(mesh_lines)

        n_line = 0
        for line in mesh_lines:
            n_line = n_line + 1
        n_char = 0
        for point in mesh_lines[0]:
            n_char = n_char + 1
        n_char = n_char - 1 # Remove newline

        mymesh = numpy.zeros((n_char, n_line), dtype=numpy.bool) 
        i_y = 0
        for line in mesh_lines:
           i_x = 0
           line = line[:-1]
           for char in line:
               mymesh[i_x, i_y] = bool(int(char))
               i_x = i_x + 1
           i_y = i_y + 1

        my_scaled_mesh = numpy.kron(mymesh, numpy.ones((self.scale, self.scale)))

        [self.np_x, self.np_y] = numpy.shape(my_scaled_mesh)

        self.solid = numpy.zeros((self.np_x, self.np_y), dtype=numpy.bool) 
        for i_y in range(0, domain.np_y):
            for i_x in range(0, domain.np_x):
                self.solid[i_x, i_y] = bool(my_scaled_mesh[i_x, i_y])

        mesh_file.close()

        self.l_x = self.np_x * self.lu_x
        self.l_y = self.np_y * self.lu_x

    def write_mesh(self, filename="mesh.dat"):
        """ Write the mesh """

        mesh_file = open(filename, 'w')
 
        for i_y in range(0, domain.np_y):
            for i_x in range(0, domain.np_x):
                if self.solid[i_x, i_y]:
                    mesh_file.write("1")
                else:
                    mesh_file.write("0")
            mesh_file.write("\n")
        
        mesh_file.close()

    def define_solid(self):
        """ Define an array of boolean where False is fluid and True is solid """

        # Cylinder coordinates
        cylinder_x = int(self.np_x / 3)
        cylinder_y = int(self.np_y / 2)
        cylinder_radius = int(self.np_y / 9)

        self.characteristic_dimension_lu = cylinder_radius * 2.
        self.characteristic_dimension = cylinder_radius * 2. * self.lu_x

        # Define a domain.np_x, domain.np_y array of boolean
        # False where there is no obstacle and True where there is one
        self.solid = numpy.fromfunction(lambda x, y: (x - cylinder_x)**2 +
                                        (y - cylinder_y)**2 < cylinder_radius**2,
                                        (self.np_x, self.np_y))

        # Wall bottom & top
        self.solid[:, 0] = True
        self.solid[:, -1] = True

    def define_solid_volvo(self):
        """ Define an array of boolean where False is fluid and True is solid """

        # Triangle coordinates
        triangle_x = 0.780           # Pointe de gauche [m]
        triangle_y = self.np_y / 2.0  # Centre
        triangle_side = 0.004        # [m]

        self.characteristic_dimension = triangle_side

        # Define a domain.np_x, domain.np_y array of boolean
        # False where there is no obstacle and True where there is one
        self.solid = None

    def initialize_velocity(self):
        """ Define the velocity on the domain
        """

        self.init_velocity = numpy.zeros((2, self.np_x, self.np_y))
        self.init_velocity[0, :, :] = self.velocity_lu * (1- self.solid[:, :])
        self.velocity = self.init_velocity

    def compute_velocity(self, lattice, f_in, explicit=False):
        """ Compute the macroscopic velocity which is an average of
            the microscopic velocities weighted by the directional
            densities (see p34 eq15)
        """

        if explicit:
            tmp = numpy.zeros((2, domain.np_x, domain.np_y))
            for i_x in range(domain.np_x):
                for i_y in range(domain.np_y):
                    for i_dir in range(lattice.dim_q):
                        tmp[:, i_x, i_y] += f_in[i_dir, i_x, i_y] * lattice.directions[i_dir, :]

            tmp[:, :, :] = tmp[:, :, :] / self.density[:, :]
            self.velocity = tmp
        else:
            self.velocity = (numpy.dot(lattice.directions.transpose(),
                                       f_in.transpose((1, 0, 2)))
                             / self.density)
        return self.velocity

    def compute_density(self, lattice, f_in, explicit=False):
        """ Calculate macroscopic densities and velocities from the lattice units.
            The macroscopic density is the sum of direction-specific fluid densities.
            (see p33 eq14)
        """

        if explicit:
            tmp = numpy.zeros((domain.np_x, domain.np_y))
            for i_dir in range(lattice.dim_q):
                tmp[:, :] += f_in[i_dir, :, :]
            self.density = tmp
        else:
            self.density = numpy.sum(f_in, axis=0)
        return self.density

    def print_infos(self):
        """ Print infos about the domain """
        print " == Infos about the domain =="
        print "Dim x, y [m]     : ", self.l_x, self.l_y
        print "dx       [m]     : ", self.lu_x
        print "Nnodes x, y, x*y : ", self.np_x, self.np_y, self.np_x * self.np_y
        print "L        [m]     : ", self.characteristic_dimension
        print " ==                        =="

    def write_hdf(self, iteration):
        """ write the solution """
        filename = 'output/data.' + str(iteration / 100).zfill(4) + '.dat'
        h5file = h5py.File(filename, "w")
        # h5file.create
        h5file.close()


class Lattice(object):
    """ Lattice definition D2Q9"""

    def __init__(self):
        self.dim_q = 9
        self.speed = 1. # 1. # in lu/ts
        # The numbering of the directions in the lattice unit is like this: (see p32)
        # 6 2 5
        # 3 0 1
        # 7 4 8
        # The direction vectors in the lattice unit are relative to the center. (see p32)
        # In 2D this leads to:
        # (-1, 1)     ( 0, 1)     ( 1, 1)
        # (-1, 0)     ( 0, 0)     ( 1, 0)
        # (-1,-1)     ( 0,-1)     ( 1,-1)
        # So with the numbering the direction array is:
        # ( (0,0) (1,0) (0,1) (-1,0) (0,-1) (1,1) (-1,1) (-1,-1) (1,-1) )
        self.directions = numpy.zeros((self.dim_q, 2), dtype=numpy.int)
        self.directions[0] = [0, 0]
        self.directions[1] = [1, 0]
        self.directions[2] = [0, 1]
        self.directions[3] = [-1, 0]
        self.directions[4] = [0, -1]
        self.directions[5] = [1, 1]
        self.directions[6] = [-1, 1]
        self.directions[7] = [-1, -1]
        self.directions[8] = [1, -1]

        # weights for each direction for the equilibrium function
        # see p35 after eq17
        # ( 1/36)     ( 1/9 )     ( 1/36)
        # ( 1/9 )     ( 4/9 )     ( 1/9 )
        # ( 1/36)     ( 1/9 )     ( 1/36)
        self.weights = numpy.zeros(self.dim_q)
        self.weights[0] = 4. / 9.
        self.weights[1] = 1. / 9.
        self.weights[2] = 1. / 9.
        self.weights[3] = 1. / 9.
        self.weights[4] = 1. / 9.
        self.weights[5] = 1. / 36.
        self.weights[6] = 1. / 36.
        self.weights[7] = 1. / 36.
        self.weights[8] = 1. / 36.

        # Bounceback boundary (see p44)
        #         6 2 5         8 4 7
        # Lattice 3 0 1 becomes 1 0 3
        #         7 4 8         5 2 6
        self.bnd_noslip = numpy.zeros((self.dim_q), dtype=numpy.int)
        self.bnd_noslip = [ 0, 3, 4, 1, 2, 7, 8, 5, 6]

        # Bounceback along y and slip along x
        #         6 2 5         7 4 8
        # Lattice 3 0 1 becomes 3 0 1
        #         7 4 8         6 2 5
        self.bnd_slip_x = numpy.zeros((self.dim_q), dtype=numpy.int)
        self.bnd_slip_x = [ 0, 1, 4, 3, 2, 8, 7, 6, 5]

        # Bounceback along x and slip along y
        #         6 2 5         5 2 6
        # Lattice 3 0 1 becomes 1 0 3
        #         7 4 8         8 4 7
        self.bnd_slip_y = numpy.zeros((self.dim_q), dtype=numpy.int)
        self.bnd_slip_y = [ 0, 3, 2, 1, 4, 6, 5, 8, 7]

        # For convenience of treating the boundary conditions, keep in a table the indicies
        # of the dimensions relative to the left, middle and right column (axis y)
        # and also the down, middle and up line (axis x)
        self.bnd_i1 = numpy.zeros(3, dtype=int)  # First column (left bnd)
        self.bnd_i1[0] = 3
        self.bnd_i1[1] = 6
        self.bnd_i1[2] = 7
        self.bnd_i2 = numpy.zeros(3, dtype=int)  # Second column (middle vertical)
        self.bnd_i2[0] = 0
        self.bnd_i2[1] = 2
        self.bnd_i2[2] = 4
        self.bnd_i3 = numpy.zeros(3, dtype=int)  # Third column (right bnd)
        self.bnd_i3[0] = 1
        self.bnd_i3[1] = 5
        self.bnd_i3[2] = 8
        self.bnd_j1 = numpy.zeros(3, dtype=int)  # First line (down bnd)
        self.bnd_j1[0] = 4
        self.bnd_j1[1] = 7
        self.bnd_j1[2] = 8
        self.bnd_j2 = numpy.zeros(3, dtype=int)  # Second line (middle horizontal)
        self.bnd_j2[0] = 0
        self.bnd_j2[1] = 1
        self.bnd_j2[2] = 3
        self.bnd_j3 = numpy.zeros(3, dtype=int)  # Third line (up bnd)
        self.bnd_j3[0] = 2
        self.bnd_j3[1] = 5
        self.bnd_j3[2] = 6

    def print_infos(self):
        """ Print infos """
        print "Dim Q = ", self.dim_q
        print "directions = ", self.directions
        print "Weights = ", self.weights
        print "BND no slip:", self.bnd_noslip
        print "Direction & weights & noslip"
        for index in range(self.dim_q):
            print self.directions[index], " | ", self.weights[index], " | ", self.bnd_noslip[index]


def compute_equilibrium_function(domain, lattice, rho, u, explicit=False):
    """ D2Q9 Equilibrium distribution function.
        see p35 eq 17 with the basic speed of the lattice
        set at c=1"""

    # Computation of ea.velocity
    if explicit:
        # Here is the explicit formulation
        ea_u = numpy.zeros((lattice.dim_q, domain.np_x, domain.np_y))
        for i_x in range(domain.np_x):
            for i_y in range(domain.np_y):
                for i_dir in range(lattice.dim_q):
                    ea_u[i_dir, i_x, i_y] = (lattice.directions[i_dir, 0] * u[0, i_x, i_y] +
                                             lattice.directions[i_dir, 1] * u[1, i_x, i_y])
    else:
        # It can be done much faster inline with numpy:
        ea_u = numpy.dot(lattice.directions, u.transpose(1, 0, 2))

    # Computation of velocity squared field (npx, npy)
    usqr = u[0]**2 + u[1]**2

    # equilibrium distribution function
    domain.feq = numpy.zeros((lattice.dim_q, domain.np_x, domain.np_y))
    for i_dir in range(lattice.dim_q):
        domain.feq[i_dir, :, :] = (rho * lattice.weights[i_dir] *
                                   (1. + 3. * ea_u[i_dir]
                                    + 4.5 * ea_u[i_dir]**2
                                    - 1.5 * usqr))
    return domain.feq


def streaming(domain, lattice, f_in, explicit=False):
    """ Streaming via the distribution function.
        Direction specific densities are moved to
        the nearest neighbor lattice nodes.
        see p36
        Periodicity is applied at all the boundaries.
    """

    f_out = numpy.zeros((lattice.dim_q, domain.np_x, domain.np_y))

    if explicit:
        # Here is the explicit formulation
        # See p36-37
        for i_x in range(domain.np_x):
            for i_y in range(domain.np_y):
                for i_dir in range(lattice.dim_q):
                    i_xn = i_x + lattice.directions[i_dir, 0]
                    # Periodicity along x
                    if i_xn < 0:
                        i_xn = domain.np_x - 1
                    elif i_xn >= domain.np_x:
                        i_xn = 0

                    i_yn = i_y + lattice.directions[i_dir, 1]
                    # Periodicity along y
                    if i_yn < 0:
                        i_yn = domain.np_y - 1
                    elif i_yn >= domain.np_y:
                        i_yn = 0

                    f_out[i_dir, i_xn, i_yn] = f_in[i_dir, i_x, i_y]
    else:
        # It can be done much faster inline with numpy:
        for i_dir in range(lattice.dim_q):
            # numpy.roll(a, shift, axis)
            # Roll array elements along a given axis. Elements that roll beyond the last position
            # are re-introduced at the first (ensuring periodicty).
            # shift: The number of places by which elements are shifted. If a tuple, then axis must
            # be a tuple of the same size, and each of the given axes is shifted by the correspond
            # ing number. If an int while axis is a tuple of ints, then the same value is used for
            # all given axes.
            f_out[i_dir, :, :] = numpy.roll(numpy.roll(f_in[i_dir, :, :],
                                                            lattice.directions[i_dir, 0],
                                                            axis=0),
                                                 lattice.directions[i_dir, 1],
                                                 axis=1)

    return f_out

def collision(domain, lattice):
    """ Perform D2Q9 collision """

    # Collision of the particles everywhere
    domain.f_n = domain.f_tmp - (domain.f_tmp - domain.feq) / domain.tau

    # Collision with the Solid part
    # Bounceback boundary (see p44)
    for i_dir in range(lattice.dim_q):
        domain.f_n[i_dir, domain.solid] = domain.f_tmp[lattice.bnd_noslip[i_dir], domain.solid]

    return domain.f_n

def initialize(domain, lattice):
    """ Initialize the domain
        Initialize rho, u, fi, fi_eq
    """

    #domain.define_solid()
    # Initialize velocity
    domain.initialize_velocity()
    # Initialize the equilibrium  distribution function with a density of one
    domain.feq = compute_equilibrium_function(domain, lattice, 1.0, domain.init_velocity)
    # Use it to initialize the distribution function
    domain.f_n = domain.feq.copy()
    domain.f_tmp = domain.feq.copy()
    # Initialize density
    domain.density = domain.compute_density(lattice, domain.f_n, explicit=False)

def apply_boundaries(domain, lattice, bnd_ux):
    """ Initialize the domain
        Initialize rho, u, fi, fi_eq
    """

    if domain.bnd_left == "perio":
       pass # Nothing to do
    elif domain.bnd_left == "wall_noslip":
       # bounce back
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, 0, :] = domain.f_n[lattice.bnd_noslip[i_dir], 0, :]
    elif domain.bnd_left == "wall_slip":
       # symmetry bounce
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, 0, :] = domain.f_n[lattice.bnd_slip_x[i_dir], 0, :]
    elif domain.bnd_left == "inlet_neq":
       # Non equilibrium extrapolation method
       # p 79 book mohamad
       # p194 book sukop
       bnd_uy = 0
       domain.density[0, :] = ((domain.f_n[2, 0, :] +
                                domain.f_n[0, 0, :] +
                                domain.f_n[4, 0, :] +
                                 2. * (domain.f_n[6, 0, :] +
                                       domain.f_n[3, 0, :] +
                                       domain.f_n[7, 0, :]))
                                 / (1. - bnd_ux))
       domain.f_tmp[1, 0, :] = domain.f_n[3, 0, :] + 2. / 3. * domain.density[0, :] * bnd_ux
       domain.f_tmp[5, 0, :] = domain.f_n[7, 0, :] + domain.feq[5, 0, :] - domain.feq[7, 0, :]
       domain.f_tmp[8, 0, :] = domain.f_n[6, 0, :] + domain.feq[8, 0, :] - domain.feq[6, 0, :]
    elif domain.bnd_left == "inlet_eq":
       # Equilibrium bc inlet 
       # Recompute density and a new equ
       # p191 book sukop
       bnd_uy = 0
       domain.density[0, :] = ((domain.f_n[2, 0, :] +
                                domain.f_n[0, 0, :] +
                                domain.f_n[4, 0, :] +
                                 2. * (domain.f_n[6, 0, :] +
                                       domain.f_n[3, 0, :] +
                                       domain.f_n[7, 0, :]))
                                 / (1. - bnd_ux))
       domain.feq = compute_equilibrium_function(domain, lattice, domain.density, domain.init_velocity)
       domain.f_tmp[5, 0, :] = domain.feq[5, 0, :]
       domain.f_tmp[1, 0, :] = domain.feq[1, 0, :]
       domain.f_tmp[8, 0, :] = domain.feq[8, 0, :]
    elif domain.bnd_left == "inlet":
       # Non equilibrium bouce back
       # Zhu He
       # p77 book Mohamad
       # p198 book Sukop
        
       bnd_uy = 0
       domain.density[0, :] = ((domain.f_n[2, 0, :] +
                                domain.f_n[0, 0, :] +
                                domain.f_n[4, 0, :] +
                                 2. * (domain.f_n[6, 0, :] +
                                       domain.f_n[3, 0, :] +
                                       domain.f_n[7, 0, :]))
                                 / (1. - bnd_ux))
       domain.f_tmp[5, 0, :] = (domain.f_n[7, 0, :]
                                + (1. / 6.) * domain.density[0, :] * bnd_ux)
       domain.f_tmp[1, 0, :] = domain.f_n[3, 0, :] + 2. / 3. * domain.density[0, :] * bnd_ux
       domain.f_tmp[8, 0, :] = (domain.f_n[6, 0, :]
                                 + (1. / 6.) * domain.density[0, :] * bnd_ux)

    if domain.bnd_right == "perio":
       pass # Nothing to do
    elif domain.bnd_right == "wall_noslip":
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, -1, :] = domain.f_n[lattice.bnd_noslip[i_dir], -1, :]
    elif domain.bnd_right == "wall_slip":
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, -1, :] = domain.f_n[lattice.bnd_slip_x[i_dir], -1, :]
    elif domain.bnd_right == "outlet":
       # Open imposed density boundary
       # p79 mohamad
       rho_outlet = 1.0
       domain.velocity[0, -1, :] = ((domain.f_n[2, -1, :] +
                                     domain.f_n[0, -1, :] +
                                     domain.f_n[4, -1, :] +
                                     2. * (domain.f_n[5, -1, :] +
                                           domain.f_n[1, -1, :] +
                                           domain.f_n[8, -1, :]))/rho_outlet - 1.)
       rhoux_out = rho_outlet * domain.velocity[0, -1, :]
       domain.f_tmp[6, -1, :] = domain.f_n[8, -1, :] - 1. / 6. * rhoux_out
       domain.f_tmp[3, -1, :] = domain.f_n[1, -1, :] - 2. / 3. * rhoux_out
       domain.f_tmp[7, -1, :] = domain.f_n[5, -1, :] - 1. / 6. * rhoux_out 
    elif domain.bnd_right == "outlet_simple":
       # Simple extrapolation at the outlet
       # p79 mohamad
       domain.f_tmp[6, -1, :] = 2. * domain.f_tmp[6, -2, :] - domain.f_tmp[6, -3, :]
       domain.f_tmp[3, -1, :] = 2. * domain.f_tmp[3, -2, :] - domain.f_tmp[3, -3, :]
       domain.f_tmp[7, -1, :] = 2. * domain.f_tmp[7, -2, :] - domain.f_tmp[7, -3, :]

    if domain.bnd_bottom == "perio":
       pass # Nothing to do
    elif domain.bnd_bottom == "wall_noslip":
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, :, 0] = domain.f_n[lattice.bnd_noslip[i_dir], :, 0]
    elif domain.bnd_bottom == "wall_slip":
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, :, 0] = domain.f_n[lattice.bnd_slip_x[i_dir], :, 0]

    if domain.bnd_up == "perio":
       pass # Nothing to do
    elif domain.bnd_up == "wall_noslip":
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, :, -1] = domain.f_n[lattice.bnd_noslip[i_dir], :, -1]
    elif domain.bnd_up == "wall_slip":
       for i_dir in range(lattice.dim_q):
           domain.f_tmp[i_dir, :, -1] = domain.f_n[lattice.bnd_slip_x[i_dir], :, -1]

    return domain.f_tmp

def time_loop(domain, lattice, inputfile):
    """ Perform the time loop """

    bnd_ux = domain.velocity_lu
    print " === Beginning iteration loop ... "
    for iteration in range(inputfile.dico['max_iter']):
        # Compute macroscopic density from f_tmp.
        domain.density = domain.compute_density(lattice, domain.f_tmp, explicit=False)

        # Compute macroscopic velocity from f_tmp.
        domain.velocity = domain.compute_velocity(lattice, domain.f_tmp, explicit=False)

        # Compute the equilibrium distribution function.
        domain.feq = compute_equilibrium_function(domain, lattice, domain.density, domain.velocity)

        # Collision of the particles from f_tmp and f_eq
        domain.f_n = collision(domain, lattice)

        # Streaming of f_n into f_tmp.
        # (Applying periodicity everywhere)
        domain.f_tmp = streaming(domain, lattice, domain.f_n)

        domain.f_tmp = apply_boundaries(domain, lattice, bnd_ux)

        #                 postproc_vel_mag = True,
        #                 postproc_density = True,
        #                 postproc_vel_ux = True)
        # Information
        it_str = "{:6.3f}".format(iteration*domain.delta_t)
        if iteration % inputfile.dico['postproc_info_niter'] == 0:
            print "- it = " + str(iteration).zfill(6) + " - t [s] =" + it_str

        # Visu output
        if iteration % inputfile.dico['postproc_dump_niter'] == 0:
            if inputfile.dico['postproc_vel_mag']:
                plt.clf()
                plt.matshow(domain.c_vel*numpy.sqrt(domain.velocity[0]**2 + domain.velocity[1]**2).transpose(),
                           cmap="jet", vmin=0, vmax=domain.c_vel*bnd_ux*1.3)
                plt.colorbar()
                plt.suptitle(' Velocity [m/s] - it ' + str(iteration).zfill(6) + ' - t [s] ' + it_str)
                output_filename = "output/vel." + str(iteration).zfill(6) + ".png"
                plt.savefig(output_filename)
                plt.close()
            if inputfile.dico['postproc_density']:
                plt.clf()
                plt.matshow(domain.density.transpose(), cmap="jet")
                plt.colorbar()
                plt.suptitle(' Density [-] - it ' + str(iteration).zfill(6) + ' - t [s] ' + it_str)
                output_filename = "output/rho." + str(iteration).zfill(6) + ".png"
                plt.savefig(output_filename)
                plt.close()
            if inputfile.dico['postproc_vel_ux']:
                plt.clf()
                plt.matshow(domain.c_vel*domain.velocity[0].transpose(), cmap="jet")
                plt.colorbar()
                plt.suptitle(' U_x [m/s] - it ' + str(iteration).zfill(6) + ' - t [s] ' + it_str)
                plt.savefig("output/ux." + str(iteration).zfill(6) + ".png")
                plt.close()

if __name__ == '__main__':
    print "============================"
    print "========== LBM 2D =========="
    print "============================"

    log = logging.getLogger(__name__)

    # Domain
    domain = Domain()
    lattice = Lattice()

    # Inputfile
    inputfile = Inputfile("inputfile")
    inputfile.apply_inputfile(domain)
    domain.velocity = (domain.viscosity * domain.reynolds) / domain.characteristic_dimension 
    domain.characteristic_dimension_lu = domain.characteristic_dimension * domain.lu_x # [lu]

    domain.read_mesh(inputfile.dico['mesh'])

    #domain.write_mesh()

    domain.delta_t = 1. / 3. * (domain.tau - 1. / 2.) * domain.lu_x * domain.lu_x / domain.viscosity # [s]
    # Law of similarity
    # Conversion factor for the velocity
    domain.c_vel = domain.lu_x / domain.delta_t
    domain.velocity_lu = domain.velocity / domain.c_vel

    print " === INFOS == "
    print " Reynolds : ", domain.reynolds
    print " Viscosity [m2/s] : ", domain.viscosity
    print " L [m] : ", domain.characteristic_dimension
    print " L [lu] : ", domain.characteristic_dimension_lu
    print " init and inlet velocity [m/s] :", domain.velocity
    print " delta_t [s] :", domain.delta_t
    print " conv_vel [m/s] : ", domain.c_vel
    print " velocity [lu/ts] :", domain.velocity_lu

    ########
    # init
    initialize(domain, lattice)

    domain.print_infos()

    # Time loop
    time_loop(domain, lattice, inputfile)
