import torch
import subprocess
import threading
import time
import os
import psutil
from datetime import datetime

class GPUMonitor:
    def __init__(self, log_file="gpu_utilization.log", interval=2.0):
        """
        Monitor GPU utilization during model training
        
        Args:
            log_file: File to save monitoring data
            interval: Monitoring interval in seconds
        """
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.thread = None
        
    def _monitor_loop(self):
        """Internal monitoring loop that runs in a separate thread"""
        with open(self.log_file, 'a') as f:
            f.write(f"Timestamp,GPU ID,GPU Name,Utilization (%),Memory Used (MB),Memory Total (MB),Temperature (C)\n")
            
        while self.running:
            try:
                # Get GPU stats using nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                     '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE, text=True, check=True
                )
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Process and log each GPU's data
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        with open(self.log_file, 'a') as f:
                            f.write(f"{timestamp},{line.strip()}\n")
                
                # Also log CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp},CPU,Usage,{cpu_percent},,,,\n")
                    f.write(f"{timestamp},RAM,Usage,{mem_percent},,,,\n")
                    
            except Exception as e:
                print(f"Error in GPU monitoring: {e}")
                
            time.sleep(self.interval)
    
    def start(self):
        """Start GPU monitoring in a background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
            print(f"GPU monitoring started. Logging to {self.log_file}")
    
    def stop(self):
        """Stop GPU monitoring"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=self.interval+1)
            print(f"GPU monitoring stopped. Log saved to {self.log_file}")
    
    def __enter__(self):
        """Support for 'with' statement"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for 'with' statement"""
        self.stop()


# Example usage
if __name__ == "__main__":
    print("GPU Monitoring Test")
    print("==================")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        
        # Start monitoring
        with GPUMonitor(interval=1.0) as monitor:
            # Simulate GPU load with a simple PyTorch operation
            print("Creating a test tensor on GPU...")
            x = torch.randn(5000, 5000, device="cuda")
            y = torch.randn(5000, 5000, device="cuda")
            
            print("Performing matrix multiplication to generate GPU load...")
            for i in range(5):
                print(f"Iteration {i+1}/5")
                z = torch.matmul(x, y)
                time.sleep(2)
                
        print("Test completed. Check gpu_utilization.log for results")
    else:
        print("CUDA is not available on this system")