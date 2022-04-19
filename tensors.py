# https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html

import torch
import numpy as np


def main():
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f'x_data:\n\t{x_data}\n')

    # alternatively,
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f'x_np:\n\t{x_np}\n')

    x_ones = torch.ones_like(x_data)
    print(f'ones tensor:\n\t{x_ones}\n')

    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f'random tensor:\n\t{x_rand}\n')

    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n\t{rand_tensor}\n")
    print(f"Ones Tensor: \n\t{ones_tensor}\n")
    print(f"Zeros Tensor: \n\t{zeros_tensor}\n")

    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    # We move our tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")

    print(f"After change, device tensor is stored on: {tensor.device}")

    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:, 1] = 0
    print(tensor)

    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # The @ symbol is matrix multiplication: https://stackoverflow.com/a/28997112
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)
    print('----- matrix multiplication -----')
    print(f'y1:\n\t{y1}\n')
    print(f'y2:\n\t{y2}\n')
    print(f'y3:\n\t{y2}\n')

    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    print('----- element-wise multiplication -----')
    print(f'z1:\n\t{y1}\n')
    print(f'z2:\n\t{z2}\n')
    print(f'z3:\n\t{z3}\n')

    # convert to single-element tensor with item()
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))

    # in-place operations are suffixed with a _
    print(f"{tensor} \n")
    tensor.add_(5)
    print(tensor)

    # Tensors on CPU and numpy arrays share memory
    # read as, operate by reference
    print('cpu and numpy')
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    print('add 1 to t in place')
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    if torch.cuda.is_available():
        # let's try with GPU tensor?
        print('cuda and numpy')
        cuda_t = torch.ones(5, device='cuda')
        print(f"cuda_t: {cuda_t}")
        try:
            cuda_n = cuda_t.numpy()
            print(f'cuda_n: {cuda_n}')
        except TypeError as e:
            print('error converting tensor to numpy:', e)

    print("let's try numpy to tensor")
    n = np.ones(5)
    t = torch.from_numpy(n)
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")


if __name__ == '__main__':
    main()
