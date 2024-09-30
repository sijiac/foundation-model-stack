import torch
from scipy.linalg import hadamard

# Function to create a random diagonal matrix with values +1 and -1
def random_diagonal_matrix(size):
    diag = torch.randint(0, 2, (size,)).float() * 2 - 1  # Generates values either +1 or -1
    return torch.diag(diag)

# Function to create the orthogonal matrix M
def construct_orthogonal_matrix(D):
    # Step 1: Create a random diagonal matrix with values +1 or -1
    diag_matrix = random_diagonal_matrix(D)
    
    # Step 2: Create a Hadamard matrix (orthogonal matrix of size D)
    H_matrix = hadamard(D)  # Using scipy's Hadamard matrix
    H_matrix_torch = torch.tensor(H_matrix, dtype=torch.float32)
    
    # Step 3: Multiply the diagonal matrix by the Hadamard matrix
    # Order of multiplication is important to ensure orthogonality
    M = torch.matmul(diag_matrix, H_matrix_torch)
    
    return M

def generate_random_orthogonal_matrix(n, seed=None):
    """
    Generates a random n x n orthogonal matrix using QR decomposition.

    Parameters:
        n (int): Size of the matrix (n x n).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: An n x n orthogonal matrix.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate a random n x n matrix
    A = torch.randn(n, n)
    
    # Perform QR decomposition
    Q, R = torch.linalg.qr(A)
    
    # Adjust Q to ensure a proper orthogonal matrix
    diag_R = torch.diag(R)
    sign_diag_R = torch.sign(diag_R)
    D = torch.diag(sign_diag_R)
    Q = Q @ D
    
    return Q

# Example dimensions
D = 8  # D must be a power of 2 for the Hadamard transform

# Construct the orthogonal matrix M
M = construct_orthogonal_matrix(D)

# Verify that M is orthogonal (M * M^T = I)
identity_check = torch.matmul(M, M.transpose(-1, -2))

breakpoint()

identity_matrix = torch.eye(D)

print("Is M orthogonal?", torch.allclose(identity_check, identity_matrix, atol=1e-6))
