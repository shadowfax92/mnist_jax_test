import os

# Configuration for CPU/GPU usage
USE_CPU_ONLY = False

def configure_xla():
    flags = os.environ.get("XLA_FLAGS", "")
    if USE_CPU_ONLY:
        flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # GPU flags
        flags += (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=false "
            "--xla_gpu_enable_async_collectives=true "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
        )
    os.environ["XLA_FLAGS"] = flags