import torch

def run_learn(*args, **kwargs):
    """Torch and Tensors"""

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    np_data = torch.randn(2, 2).numpy()
    tensor = torch.from_numpy(np_data).to(torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"DataType: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires Grad: {tensor.requires_grad}")
    print(f"Is CUDA: {tensor.is_cuda}")
