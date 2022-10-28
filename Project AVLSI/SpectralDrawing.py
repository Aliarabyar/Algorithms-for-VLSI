import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import scipy.sparse
import numpy as np
import networkx as nx
from scipy.io import mmread
import subprocess


# ALGORITHMS FOR VLSI
# Spectral Graph Drawing
# Ali A.Yarmohammadi
# January  2022

# the implementation of the optimized degree-normalized method,
# which is an optimization process in the Spectral method and based on
# the power-iteration method, was incomplete, unfortunately.

# After a long debugging process, the correct results (second and third eigenvectors)
# were not obtained. Implementation of the two methods Laplacian and Normalized-Laplacian
# were successful, and in order to be able to compare the methods and complete the project,
# "Spring layout," which is a Force-directed graph drawing, is used by calling the function
# called "nx.layout.spring\_layout(G)" in the Networkx package. This function positions nodes
# using Fruchterman-Reingold force-directed algorithm.

# As I couldn't finish implementing the optimized version of the algorithm,
# drawing graphs from the provided links (website) will take a couple of minutes,
# depending on how big the graph is.


def power_Iteration_degree_normalized_Koren_layout(G, eps): # incomplete

    # Returns the graph adjacency matrix as a SciPy sparse matrix.
    A = nx.to_scipy_sparse_matrix(G, format='csr')
    A = np.array(A.todense())
    n, m = A.shape

    uk_hat = np.random.random((n,1))    # Intialized vectors (second Vec)
    norm = np.linalg.norm(uk_hat)
    uk_hat = uk_hat / norm


    # For D-orthonormalization

    firstVec = np.ones(n)
    degrees = [val for (node, val) in G.degree()]
    firstVecD = degrees * firstVec
    mult1_denom = firstVec * firstVecD

    residual = np.zeros(n)
    num_iterations1 = 0

    while (True):
        uk = uk_hat
        # D-orthonormalize
        mult1_num =  uk * firstVecD
        uk = uk - (mult1_num / mult1_denom) * firstVec

        #  Do matrix-vector product

        uk_hat =  G @ uk
        norm = np.linalg.norm(uk_hat)
        uk_hat = uk_hat / norm

        num_iterations1 += 1 # to know how many iteration we have or to limit the foorLoop if needed.

        residual_dot = np.dot(uk , uk_hat)     #np.dot(uk , uk_hat)
        if (abs(residual_dot.max())>= (1 - eps)):
            break

        residual = uk - uk_hat
        residual_norm = np.linalg.norm(residual)

        if (residual_norm < eps):
         break


    # Save this eigenvector
    secondVec = uk_hat


    # # second iteration (dim=2)
    # eps = 2.0 * eps
    # # For D-orthonormalization
    # degrees = [val for (node, val) in G.degree()]
    # secondVecD =  G.degree *  secondVec
    # mult2_denom = secondVec * secondVecD
    # # Initialized vectors are passed to function
    # thirdVec = np.random(n)
    # uk_hat = thirdVec
    #
    # num_iterations2 = 5
    #
    # while (True):
    #     uk = uk_hat
    #     # D-orthonormalize
    #     mult1_num = uk * firstVecD
    #     uk = uk - (mult1_num / mult1_denom) * firstVec
    #     mult2_num = uk * secondVecD
    #     uk = uk - (mult2_num / mult2_denom) * secondVec
    #
    #     # Do matrix-vector product
    #     matrix = nx.to_numpy_matrix(G)
    #     uk_hat =np.dot(matrix , uk)
    #     np.linalg.norm(uk_hat)
    #     num_iterations1 += 1
    #
    #     residual_dot = uk * uk_hat
    #     if (residual_dot >= (1 - eps)):
    #       break
    #
    #     residual = uk - uk_hat
    #     residual_norm = np.linalg.norm(residual)
    #
    #     if (residual_norm < eps):
    #      break
    #
    #
    # # Save eigenvector
    # thirdVec = uk_hat

    return secondVec #, thirdVec



def laplacianLayout(G, normalize=False):
    # Returns the graph adjacency matrix as a SciPy sparse matrix.
    A = nx.to_scipy_sparse_matrix(G, format='csr')
    A = np.array(A.todense())
    n, m = A.shape

    # Extract a diagonal or construct a diagonal array
    D = np.diag(A.sum(1).flatten())
    L = D - A
    # print(L)

    # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    B = np.eye(n)

    if normalize:
       B = D
       # print(B)

    layout = getEigPos(G, L, B)

    return layout


def getEigPos(G, L, B):
    # Solve a standard or generalized eigenvalue problem.
    eigenvalues, eigenvectors = scipy.linalg.eigh(L, B)
    k = 2
    index = np.argsort(eigenvalues)[1:k + 1]    # 0 index is zero eigenvalue
    pos = np.real(eigenvectors[:, index])
    # Scale to std = 1
    pos = dict(zip(G, pos)) # Convert two lists into a dictionary
    return pos


def calEdgeLength(G, pos):
    len = 0
    for u, v in G.edges:
        len += np.linalg.norm(pos[u] - pos[v])
    return len


def exportToFile(G, fineName, name, title, pos=None):

    g = nx.draw_networkx_nodes(G, pos=pos, node_size=20, alpha=1, label=None, cmap=plt.cm.jet,
                               node_color=np.array([G.degree[v] for v in G.nodes()]))

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.1)

    # Set the plot and save to file using "matplotlib"
    cbar = plt.colorbar(g)
    cbar.ax.set_ylabel('Vertex Degree')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("results/" + name + "/" + fineName)
    plt.close()



def scaling(G, pos):
    A = np.array([list(e) for e in pos.values()])
    # Compute the standard deviation along the specified axis.
    std = np.std(A, axis=0)
    for u in G:
        pos[u] /= std


def drawGraph(G, name):

    print('---------------------------------------------')
    print('Drawing graph: ' + name + '...')

    # perform some drawing methods and return the layouts
    laplacian_Layout = laplacianLayout(G)
    norm_laplacian_Layout = laplacianLayout(G, normalize=True)
    force_directed_Layout = nx.layout.spring_layout(G)
    # pectral_Layout= power_Iteration_degree_normalized_Koren_layout(G,  eps=1e-9)



    # scaling graphs
    scaling(G, laplacian_Layout)
    scaling(G, norm_laplacian_Layout)
    scaling(G, force_directed_Layout)
    # scaling(G, Spectral_Layout)

    print(f"laplacian edge length                : {calEdgeLength(G, laplacian_Layout):.1f}")
    print(f'normalized laplacian edge length     : {calEdgeLength(G, norm_laplacian_Layout):.1f}')
    print(f"force_directed edge length           : {calEdgeLength(G, force_directed_Layout):.1f}")
    # print("Spectral power-iteration edge_length : {:.5f}".format(calEdgeLength(G, Spectral_Layout)))

    # calculation of edgelength
    edgeLen_laplacian = calEdgeLength(G, laplacian_Layout)
    edgeLen_norm = calEdgeLength(G, norm_laplacian_Layout)
    edgeLen_force_directed = calEdgeLength(G, force_directed_Layout)
    # edgeLen_Spectral = caledgelength(G, Spectral_Layout)

    # defining chart's label
    laplacian_title = f'method: Laplacian, edge length {edgeLen_laplacian:.1f}'
    norm_laplacian_title = f' method: Normalized Laplacian, edge length {edgeLen_norm:.1f}'
    force_directed_title = f'method: Force_directed (spring), edge length {edgeLen_force_directed:.1f}'
    # Spectral_title = 'method: Spectral_Koren , edgelength {:.1f}'.format(edgeLen_Spectral)

    # export results to the folder "/results/"
    exportToFile(G, 'Laplacian.jpg', name, title=laplacian_title, pos=laplacian_Layout)
    exportToFile(G, 'Normalized-Laplacian.jpg', name, title=norm_laplacian_title, pos=norm_laplacian_Layout)
    exportToFile(G, 'Force_directed.jpg', name, title=force_directed_title, pos=force_directed_Layout)
    # exportToFile(G, 'Spectral_Koren.jpg', name, title=Spectral_title, pos=Spectral_Layout)

    print('Results were saved in:   ' + 'results/' + name)
    print('---------------------------------------------')

def loadMatrix(filename):
    M = mmread('Benchmark_yifanhuNetGALLERY/' + filename)
    if scipy.sparse.issparse(M):
        M = M.todense()
    return M





def main():
    # defining graphs

    # Creat an Erdos_Renyi random graph with 200 nodes and the probability for edge creation=0.15
    G = nx.erdos_renyi_graph(200, 0.15, seed=1, directed=False)
    drawGraph(G, "randomGraph")


    ###  BENCHMARKS

    ### Graph visualization of matrices from the University of Florida Collection
    ### Sample graphs were selected from the first page: http://yifanhu.net/GALLERY/GRAPHS/
    ###  Drawing of graphs from the provided links will take a couple of minutes depending
    # on how big is the graph and the processing power of the computer.

    ### Please replace the file name in the "loadMatrix()" function below.

    ### ----------------------------------------------------------------------------------------
    ### File File Name:  3elt.mtx      Link: http://yifanhu.net/GALLERY/GRAPHS/GIF_SMALL/AG-Monien@3elt.html
    ### File File Name:  diag.mtx      Link: https://www.cise.ufl.edu/research/sparse/matrices/AG-Monien/diag.html
    ### File File Name:  grid1.mtx     Link: https://www.cise.ufl.edu/research/sparse/matrices/AG-Monien/grid1.html
    ### File File Name:  grid2.mtx     Link: http://yifanhu.net/GALLERY/GRAPHS/GIF_SMALL/AG-Monien@grid2.html
    ### ----------------------------------------------------------------------------------------

    G = nx.Graph(loadMatrix("grid1.mtx"))  # The file name can be chosen from the lines above.
    drawGraph(G, "yifanhu.net")




if __name__ == "__main__":
    main()
