from simplicial import *
from random import Random
from typing import List
from sympy import Matrix
import numpy as np
import galois


##########################################################################
# General utils

GF2 = galois.GF(2)
rand = Random()
get_triangles = lambda X:X.allSimplices(lambda X,s:X.orderOf(s)==2)
get_edges = lambda X:X.allSimplices(lambda X,s:X.orderOf(s)==1)
get_vertices = lambda X:X.allSimplices(lambda X,s:X.orderOf(s)==0)
get_n_vertices = lambda X:len(get_vertices(X))
##########################################################################



##########################################################################
# Complex methods

def generate_random_complex(n: int)->SimplicialComplex:
    """
    Generates a complex with n 0-simplices and random triangles.
    """
    X = SimplicialComplex()
    for i in range(n):
        X.addSimplex(id=i)

    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1, n):
                if rand.choice([True,False]):
                    X.addSimplexWithBasis(bs=[i,j,k], id=(i,j,k))

    for simp in get_edges(X):
        X.relabelSimplex(simp, tuple(X.boundary([simp])))
    return X


def get_euler_cycle_basis(X: SimplicialComplex):
    """
    Returns a basis for the space of euler cycles of a complex.
    The space is defined as {A\in p([n]^3): Boundry(A)=0}.
    In words, return a basis for the space of 2-chains with empty boundary.
    """
    # Create matrix with column per triangle and row per edge, a_ij=1 iff i in boundary(j)
    boundaries = X.boundaryOperator(2)
    print("boundary")

    gf2_boundaries = GF2(boundaries)
    gf2_null_basis = GF2.null_space(gf2_boundaries)

    #Turn each null basis element to a linear combination of triangles.
    triangles = get_triangles(X)
    n_triangles = len(triangles)
    boundary_basis = list()
    for i in range(len(gf2_null_basis)):
        vec = gf2_null_basis[i]
        two_chain = [triangles[j] for j in range(n_triangles) if vec[j]]
        boundary_basis.append(two_chain)
    return boundary_basis

def get_cuts_basis(X: SimplicialComplex):
    """
    Returns a basis for the cuts.
    The space is defined as {co-boundary(A): A\in p([n]^2)}
    In words: it is the set of coboundaries of all edge sets.
    The basis is the set of coboundaries of all singleton 1-chains.
    """
    cuts_basis = list()
    for edge in get_edges(X):
        coboundary = [tr for tr in get_triangles(X) if edge in X.boundary([tr])]
        print("coboundary")
        cuts_basis.append(coboundary)
    return cuts_basis

##########################################################################


##########################################################################
# Chain list parsing


def two_chain_to_vector(X: SimplicialComplex, two_chain: List[Simplex])->np.ndarray:
    """
    Gets a list of triangles and returns a 1-column matrix with corresponding values.
    """
    X_triangles = get_triangles(X)
    n_triangles = len(X_triangles)
    vec = [[1 if X_triangles[i] in two_chain else 0] for i in range(n_triangles)]
    return np.array(vec,dtype=int)

def two_chain_list_to_matrix(X: SimplicialComplex, two_chain_list: List[List[Simplex]])->np.ndarray:
    if len(two_chain_list) == 0:
        return np.ndarray((len(get_triangles(X)),0),dtype=int)
    vectors = (two_chain_to_vector(X,two_chain_list[i]) for i in range(len(two_chain_list)))
    return np.hstack(tuple(vectors))

##########################################################################



##########################################################################
# Gallai cycle-co-cycle test

def is_applying_gallai_theorem(X: SimplicialComplex)->bool:
    euler = get_euler_cycle_basis(X)
    euler_mat = two_chain_list_to_matrix(X,euler)
    cut = get_cuts_basis(X)
    cut_mat = two_chain_list_to_matrix(X,cut)

    space_sum_mat = np.hstack((euler_mat,cut_mat))
    all_triangles_vec = np.array([[1] for i in range(len(get_triangles(X)))])
    combined_mat = np.hstack((space_sum_mat,all_triangles_vec))

    space_sum_mat = GF2(space_sum_mat)
    combined_mat = GF2(space_sum_mat)
    space_sum_rank = len(space_sum_mat.column_space())
    combined_rank = len(combined_mat.column_space())

    if space_sum_rank < len(get_triangles(X))-4:
        print("space sum is much lower dim than the entire space")
        print(space_sum_rank)
        print(len(get_triangles(X)))

    return space_sum_rank == combined_rank

##########################################################################


for i in range(1):
    X = generate_random_complex(16)
    if not is_applying_gallai_theorem(X):
        print(i)
        print(f'edges: {get_edges(X)}')
        print(f'vertices: {get_vertices(X)}')
        print(f'triangles: {get_triangles(X)}')
        print("Simulation showed")
        break
print("finished")

# print(f'euler: {euler}')
# print(f'cuts: {cut}')
# print(f'euler mat: {euler_mat}')
# print(f'cut mat: {cut_mat}')
# print(f'both: {space_sum_mat}')
# print(f'degree: {space_sum_rank}')
# print(f'all: {combined_mat}')
# print(f'degree: {combined_rank}')
