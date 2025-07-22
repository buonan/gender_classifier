import torch
import os

def print_cuda_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")

if __name__ == "__main__":
    print_cuda_info()