import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import warnings

# Add check content
def _to_coo_float32(mat):
    """Convert input to float32-COO; if already sparse then directly astype"""
    if sp.issparse(mat):
        return mat.astype(np.float32).tocoo()
    return sp.coo_matrix(np.asarray(mat, dtype=np.float32))

def _fix_invalid(spmat):
    """If sparse matrix data contains NaN/Inf, replace it with 0."""
    if np.isnan(spmat.data).any() or np.isinf(spmat.data).any():
        spmat.data = np.nan_to_num(spmat.data, nan=0.0, posinf=0.0,
                                   neginf=0.0)
    return spmat

def normalize_adj_mx(adj_mx, adj_type, return_type='dense'):
    """
    Normalize adjacency matrix
    
    Args:
        adj_mx: Adjacency matrix
        adj_type: Normalization type ('normlap', 'scalap', 'symadj', 'transition', 'doubletransition', 'identity')
        return_type: Return type ('dense', 'coo')
        
    Returns:
        List of normalized adjacency matrices
    """
    adj_mx = _to_coo_float32(adj_mx) 
    adj_mx = _fix_invalid(adj_mx)
    
    if adj_type == 'normlap':
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == 'symadj':
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == 'transition':
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == 'doubletransition':
        # Always return two matrices - if one fails use identity matrix as placeholder
        try:
            adj_f = calculate_asym_adj(adj_mx)
        except Exception as e:
            warnings.warn(f'Forward failed: {e}, using I as placeholder')
            adj_f = sp.eye(adj_mx.shape[0], format='coo', dtype=np.float32)
        try:
            adj_b = calculate_asym_adj(adj_mx.T)
        except Exception as e:
            warnings.warn(f'Backward failed: {e}, using I as placeholder')
            adj_b = sp.eye(adj_mx.shape[0], format='coo', dtype=np.float32)
        adj = [adj_f, adj_b]
    elif adj_type == 'identity':
        adj = [sp.eye(adj_mx.shape[0], format='coo', dtype=np.float32)]
    else:
        warnings.warn(f"Unknown adjacency matrix type: {adj_type}")
        return []

    # 3) Output format
    if return_type == 'dense':
        return [a.todense() for a in adj]
    else:  # 'coo'
        return [_fix_invalid(a).tocoo() for a in adj]


def calculate_normalized_laplacian(adj_mx):
    """
    Calculate normalized Laplacian matrix L = I - D^(-1/2) * A * D^(-1/2)
    Handle zero-degree node issues
    """
    A = _to_coo_float32(adj_mx)
    deg = np.array(A.sum(1)).flatten()
    inv_sqrt = np.power(deg, -0.5, where=deg > 1e-10)
    D_inv_sqrt = sp.diags(inv_sqrt)
    L = sp.eye(A.shape[0], dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    return _fix_invalid(L.tocoo())


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    """
    Calculate scaled Laplacian matrix
    Provide fallback if eigenvalue calculation fails
    """
    A = _to_coo_float32(adj_mx)
    if undirected:
        A = sp.coo_matrix(np.maximum(A.toarray(), A.T.toarray()), dtype=np.float32)
    L = calculate_normalized_laplacian(A)
    if lambda_max is None:
        try:
            lambda_max = linalg.eigsh(L, 1, which='LM', return_eigenvectors=False)[0]
            if not np.isfinite(lambda_max) or lambda_max <= 0:
                raise ValueError
        except Exception:
            warnings.warn("lambda_max estimation failed, using 2.0")
            lambda_max = 2.0
    I = sp.eye(L.shape[0], dtype=np.float32)
    L_scaled = (2.0 / lambda_max) * L - I
    return _fix_invalid(L_scaled.tocoo())


def calculate_sym_adj(adj_mx):
    """
    Calculate symmetric normalized adjacency matrix A' = D^(-1/2) * A * D^(-1/2)
    Handle zero-degree node issues
    """
    A = _to_coo_float32(adj_mx)
    deg = np.array(A.sum(1)).flatten()
    inv_sqrt = np.power(deg, -0.5, where=deg > 1e-10)
    D_inv_sqrt = sp.diags(inv_sqrt)
    return _fix_invalid((D_inv_sqrt @ A @ D_inv_sqrt).tocoo())


def calculate_asym_adj(adj_mx):
    """
    Calculate asymmetric normalized adjacency matrix (transition probability matrix) A' = D^(-1) * A
    Handle zero-degree node issues
    """
    A = _to_coo_float32(adj_mx)
    deg = np.array(A.sum(1)).flatten()
    inv = np.power(deg, -1.0, where=deg > 1e-10)
    D_inv = sp.diags(inv)
    return _fix_invalid((D_inv @ A).tocoo())


def calculate_cheb_poly(L, Ks):
    """
    Calculate Chebyshev polynomials
    
    Args:
        L: Scaled Laplacian matrix
        Ks: Polynomial order
        
    Returns:
        Chebyshev polynomial coefficients
    """
    # Ensure input matrix is valid
    if np.isnan(L).any() or np.isinf(L).any():
        warnings.warn("Input Laplacian matrix contains invalid values, attempting to fix")
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
    
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    
    # Recursively calculate Chebyshev polynomials
    for i in range(2, Ks):
        try:
            LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
            
            # Check intermediate results
            if np.isnan(LL[-1]).any() or np.isinf(LL[-1]).any():
                warnings.warn(f"Invalid values appeared in Chebyshev polynomial calculation (order {i}), attempting to fix")
                LL[-1] = np.nan_to_num(LL[-1], nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            warnings.warn(f"Error calculating Chebyshev polynomial (order {i}): {str(e)}")
            # If calculation fails, use previous order polynomial as substitute
            LL.append(LL[-1])
    
    return np.asarray(LL)
