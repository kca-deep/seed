# comprehensive_check.py
import subprocess
import sys
import platform


def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Error"


def check_environment():
    print("=== 종합 환경 확인 ===\n")

    # 1. 시스템 정보
    print("1. 시스템 정보")
    print(f"  - OS: {platform.platform()}")
    print(f"  - Python: {platform.python_version()}")
    print(f"  - Architecture: {platform.architecture()[0]}")
    print()

    # 2. GPU 정보
    print("2. GPU 정보")
    nvidia_smi = run_command(
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
    )
    if nvidia_smi != "Error":
        print(f"  - GPU: {nvidia_smi}")
    else:
        print("  - GPU: Not available")
    print()

    # 3. CUDA 정보
    print("3. CUDA 정보")
    cuda_version = run_command(
        "cat /usr/local/cuda/version.txt 2>/dev/null || echo 'Not found'"
    )
    print(f"  - CUDA: {cuda_version}")
    cudnn_version = run_command(
        "cat /usr/local/cuda/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2 || echo 'Not found'"
    )
    print(f"  - cuDNN: {cudnn_version if 'Not found' in cudnn_version else 'Found'}")
    print()

    # 4. Python 패키지
    print("4. 주요 Python 패키지")
    packages = {
        "tensorflow": "TensorFlow",
        "torch": "PyTorch",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "scipy": "SciPy",
        "matplotlib": "Matplotlib",
        "scikit-learn": "Scikit-learn",
    }

    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "Unknown version")
            print(f"  - {display_name}: {version}")
        except ImportError:
            print(f"  - {display_name}: Not installed")
    print()

    # 5. TensorFlow GPU 테스트
    print("5. TensorFlow GPU 테스트")
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        print(f"  - GPU 디바이스 수: {len(gpus)}")
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu}")
        else:
            print("  - GPU 사용 불가")
    except Exception as e:
        print(f"  - TensorFlow error: {e}")


if __name__ == "__main__":
    check_environment()
