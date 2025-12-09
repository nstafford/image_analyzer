"""System capability detection for ML model loading."""

import platform
from typing import Dict, Any
import subprocess


def get_system_info() -> Dict[str, Any]:
    """
    Detect system capabilities including RAM, GPU, and VRAM.
    
    Returns:
        Dictionary containing system information
    """
    system_info = {
        'platform': platform.system(),
        'ram_total_mb': 0,
        'ram_available_mb': 0,
        'gpu_available': False,
        'gpu_name': None,
        'gpu_vram_mb': 0,
        'cuda_available': False,
        'recommended_device': 'cpu'
    }
    
    # Check RAM
    try:
        if platform.system() == 'Windows':
            system_info.update(_get_windows_memory())
        elif platform.system() == 'Linux':
            system_info.update(_get_linux_memory())
        elif platform.system() == 'Darwin':  # macOS
            system_info.update(_get_macos_memory())
    except Exception as e:
        print(f"Error detecting RAM: {e}")
    
    # Check GPU and CUDA
    try:
        import torch
        system_info['cuda_available'] = torch.cuda.is_available()
        
        if system_info['cuda_available']:
            system_info['gpu_available'] = True
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            # Get VRAM in MB
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            system_info['gpu_vram_mb'] = gpu_memory / (1024 ** 2)
            system_info['recommended_device'] = 'cuda'
        else:
            # Check for other GPU types (Intel, AMD, MPS for Apple Silicon)
            if platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                system_info['gpu_available'] = True
                system_info['gpu_name'] = 'Apple Silicon GPU'
                system_info['recommended_device'] = 'mps'
            else:
                # Try to detect non-CUDA GPUs on Windows
                if platform.system() == 'Windows':
                    gpu_info = _get_windows_gpu()
                    system_info.update(gpu_info)
    except ImportError:
        print("PyTorch not installed - cannot detect CUDA")
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    return system_info


def _get_windows_memory() -> Dict[str, int]:
    """Get memory information on Windows."""
    try:
        result = subprocess.run(
            ['wmic', 'OS', 'get', 'TotalVisibleMemorySize,FreePhysicalMemory', '/value'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.strip().split('\n')
        memory_info = {}
        
        for line in lines:
            if '=' in line:
                key, value = line.split('=')
                if key.strip() == 'TotalVisibleMemorySize':
                    memory_info['ram_total_mb'] = int(value.strip()) // 1024
                elif key.strip() == 'FreePhysicalMemory':
                    memory_info['ram_available_mb'] = int(value.strip()) // 1024
        
        return memory_info
    except Exception as e:
        print(f"Error getting Windows memory: {e}")
        return {'ram_total_mb': 0, 'ram_available_mb': 0}


def _get_windows_gpu() -> Dict[str, Any]:
    """Get GPU information on Windows."""
    try:
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM', '/value'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.strip().split('\n')
        gpu_name = None
        vram_bytes = 0
        
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                if key.strip() == 'Name' and value.strip():
                    gpu_name = value.strip()
                elif key.strip() == 'AdapterRAM' and value.strip():
                    vram_bytes = int(value.strip())
        
        return {
            'gpu_available': gpu_name is not None,
            'gpu_name': gpu_name,
            'gpu_vram_mb': vram_bytes / (1024 ** 2) if vram_bytes > 0 else 0
        }
    except Exception as e:
        print(f"Error getting Windows GPU: {e}")
        return {'gpu_available': False, 'gpu_name': None, 'gpu_vram_mb': 0}


def _get_linux_memory() -> Dict[str, int]:
    """Get memory information on Linux."""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        
        memory_info = {}
        for line in lines:
            if line.startswith('MemTotal:'):
                memory_info['ram_total_mb'] = int(line.split()[1]) // 1024
            elif line.startswith('MemAvailable:'):
                memory_info['ram_available_mb'] = int(line.split()[1]) // 1024
        
        return memory_info
    except Exception as e:
        print(f"Error getting Linux memory: {e}")
        return {'ram_total_mb': 0, 'ram_available_mb': 0}


def _get_macos_memory() -> Dict[str, int]:
    """Get memory information on macOS."""
    try:
        # Get total memory
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True, timeout=5)
        total_bytes = int(result.stdout.split(':')[1].strip())
        
        # Get available memory (rough estimate)
        result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.split('\n')
        free_pages = 0
        
        for line in lines:
            if 'Pages free' in line:
                free_pages = int(line.split(':')[1].strip().rstrip('.'))
                break
        
        page_size = 4096  # typical page size
        available_bytes = free_pages * page_size
        
        return {
            'ram_total_mb': total_bytes // (1024 ** 2),
            'ram_available_mb': available_bytes // (1024 ** 2)
        }
    except Exception as e:
        print(f"Error getting macOS memory: {e}")
        return {'ram_total_mb': 0, 'ram_available_mb': 0}


def format_system_info(system_info: Dict[str, Any]) -> str:
    """
    Format system information for display.
    
    Args:
        system_info: Dictionary from get_system_info()
        
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"Platform: {system_info['platform']}")
    lines.append(f"RAM: {system_info['ram_total_mb']} MB total, {system_info['ram_available_mb']} MB available")
    
    if system_info['gpu_available']:
        lines.append(f"GPU: {system_info['gpu_name']}")
        if system_info['gpu_vram_mb'] > 0:
            lines.append(f"VRAM: {system_info['gpu_vram_mb']:.0f} MB")
        lines.append(f"CUDA Available: {'Yes' if system_info['cuda_available'] else 'No'}")
    else:
        lines.append("GPU: None detected")
    
    lines.append(f"Recommended Device: {system_info['recommended_device'].upper()}")
    
    return '\n'.join(lines)
