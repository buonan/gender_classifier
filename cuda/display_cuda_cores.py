import subprocess
import torch
import json
import re

def get_gpu_info():
    """Get detailed GPU information including CUDA cores if available"""
    print("PyTorch GPU Information:")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("CUDA is not available on this system")
    
    print("\nDetailed NVIDIA GPU Information:")
    try:
        # Try to get detailed information using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=gpu_name,driver_version,memory.total,tcc_driver', '--format=csv,noheader'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        print(result.stdout)
        
        # Try to get CUDA core count using nvidia-smi with GPU query
        try:
            # This approach works on some systems
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,cuda_cores', '--format=csv,noheader'],
                stdout=subprocess.PIPE, text=True, check=True
            )
            if 'cuda_cores' in result.stdout and not re.search(r'[Nn]/[Aa]', result.stdout):
                print("\nCUDA Cores Information:")
                print(result.stdout)
            else:
                raise subprocess.SubprocessError("CUDA cores not directly available")
        except (subprocess.SubprocessError, FileNotFoundError):
            # Alternative approach using device properties
            print("\nCUDA Cores Information:")
            print("Note: Exact CUDA core count may not be directly available through nvidia-smi")
            print("The core count depends on the GPU architecture and compute capability")
            
            # Try to get architecture information
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
                    stdout=subprocess.PIPE, text=True, check=True
                )
                print(result.stdout)
                print("\nTo find the exact CUDA core count, look up your GPU model and compute capability")
                print("in NVIDIA's documentation or use third-party tools like 'deviceQuery' from CUDA samples.")
            except:
                print("Could not retrieve compute capability information")
    
    except (subprocess.SubprocessError, FileNotFoundError):
        print("nvidia-smi is not available. Make sure NVIDIA drivers are properly installed.")
        
        # Try lspci as a fallback
        try:
            print("\nTrying lspci to detect GPUs:")
            result = subprocess.run(['lspci', '|', 'grep', '-i', 'nvidia'], 
                                   shell=True, stdout=subprocess.PIPE, text=True)
            print(result.stdout)
        except:
            print("Could not detect GPUs using lspci")

if __name__ == "__main__":
    print("CUDA Cores Information Tool")
    print("===========================")
    get_gpu_info()
    
    print("\nNote: The exact number of CUDA cores depends on the GPU architecture.")
    print("For consumer GPUs (GeForce), NVIDIA typically reports CUDA cores directly.")
    print("For professional GPUs (Tesla, Quadro), NVIDIA may report streaming multiprocessors (SMs).")
    print("You can calculate CUDA cores as: Number of SMs × Cores per SM for your specific architecture.")