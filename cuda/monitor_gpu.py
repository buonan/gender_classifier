import subprocess
import time
import argparse
import os

def monitor_gpu(interval=1, duration=None):
    """
    Monitor GPU usage using nvidia-smi
    
    Args:
        interval: Time between updates in seconds
        duration: Total monitoring duration in seconds (None for indefinite)
    """
    # Check if nvidia-smi is available
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: nvidia-smi not found. Make sure NVIDIA drivers are installed correctly.")
        return
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Run nvidia-smi with formatting for better readability
            subprocess.run([
                "nvidia-smi", 
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader"
            ])
            
            # Also show processes using GPUs
            print("\nProcesses using GPUs:")
            subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
            
            iteration += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            print(f"\nMonitoring for {elapsed:.1f}s (Refresh: {interval}s, Iterations: {iteration})")
            print("Press Ctrl+C to stop monitoring")
            
            # Check if monitoring duration has been reached
            if duration and elapsed >= duration:
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nGPU monitoring stopped by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor NVIDIA GPU usage")
    parser.add_argument("-i", "--interval", type=float, default=2.0, 
                        help="Update interval in seconds (default: 2.0)")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Total monitoring duration in seconds (default: indefinite)")
    args = parser.parse_args()
    
    monitor_gpu(interval=args.interval, duration=args.duration)