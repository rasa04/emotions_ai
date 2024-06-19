import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
from pynvml import *
from rich.console import Console
from rich.table import Table

from model import predict_emotions

# Initialize NVML for GPU monitoring
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assuming you have one GPU

REQUESTS_AMOUNT = 1000

# Initialize rich console
console = Console()


# Function to generate random text
def generate_random_text(length=100):
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + " ", k=length))


# Function to measure the response time and memory usage of the model
def measure_response_time_and_memory(request_text: str):
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    request_start_time = time.time()
    _, handled_input_tokens, handled_output_tokens = predict_emotions(request_text)
    return (
        time.time() - request_start_time,
        max(process.memory_info().rss - initial_memory, 0),  # Ensure memory used is not negative
        handled_input_tokens,
        handled_output_tokens
    )


# Function to get CPU and GPU usage
def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.used / (1024 ** 2)  # Convert to MB

    gpu_util = nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_usage = gpu_util.gpu
    gpu_memory_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_memory_usage = gpu_memory_info.used / (1024 ** 2)  # Convert to MB

    return cpu_usage, memory_usage, gpu_usage, gpu_memory_usage


# Print system metrics in real-time
def print_system_metrics():
    while True:
        cpu_usage, memory_usage, gpu_usage, gpu_memory_usage = get_system_metrics()
        console.print(f"CPU Usage: {cpu_usage:.2f}%, Memory Usage: {memory_usage:.2f} MB, "
                      f"GPU Usage: {gpu_usage:.2f}%, GPU Memory Usage: {gpu_memory_usage:.2f} MB", style="bold magenta")
        time.sleep(1)


# Start real-time metrics printing in a separate thread
metrics_thread = threading.Thread(target=print_system_metrics)
metrics_thread.daemon = True
metrics_thread.start()

# Single-threaded load test
console.print("Running single-threaded load test...", style="bold blue")
single_thread_response_times = []
single_thread_memory_usages = []
single_thread_input_tokens = []
single_thread_output_tokens = []
start_time = time.time()
for _ in range(REQUESTS_AMOUNT):
    text = generate_random_text()  # Generate random text
    response_time, memory_used, input_tokens, output_tokens = measure_response_time_and_memory(text)
    single_thread_response_times.append(response_time)
    single_thread_memory_usages.append(memory_used)
    single_thread_input_tokens.append(input_tokens)
    single_thread_output_tokens.append(output_tokens)
end_time = time.time()
single_thread_duration = end_time - start_time
console.print("Single-threaded load test completed.", style="bold green")

# Multithreaded load test
console.print("Running multi-threaded load test...", style="bold blue")
multi_thread_response_times = []
multi_thread_memory_usages = []
multi_thread_input_tokens = []
multi_thread_output_tokens = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    for future in [
        executor.submit(
            measure_response_time_and_memory,
            generate_random_text()
        ) for _ in range(REQUESTS_AMOUNT)
    ]:
        try:
            response_time, memory_used, input_tokens, output_tokens = future.result()
            multi_thread_response_times.append(response_time)
            multi_thread_memory_usages.append(memory_used)
            multi_thread_input_tokens.append(input_tokens)
            multi_thread_output_tokens.append(output_tokens)
        except Exception as e:
            console.print(f"Error: {e}", style="bold red")
end_time = time.time()
multi_thread_duration = end_time - start_time
console.print("Multi-threaded load test completed.", style="bold green")


# Print summary statistics
def print_summary(response_times, memory_usages, input_tokens, output_tokens, duration, label):
    table = Table(title=f"{label} Load Test Results")
    table.add_column("Statistic", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value (seconds / bytes)", justify="right", style="magenta")
    table.add_column("Value (ms / MB)", justify="right", style="green")

    if response_times:
        table.add_row("Total Duration (seconds)", f"{duration:.2f}", f"{duration * 1000:.2f} ms")
        table.add_row("Average Response Time (seconds)", f"{statistics.mean(response_times):.4f}",
                      f"{statistics.mean(response_times) * 1000:.2f} ms")
        table.add_row("Median Response Time (seconds)", f"{statistics.median(response_times):.4f}",
                      f"{statistics.median(response_times) * 1000:.2f} ms")
        table.add_row("Maximum Response Time (seconds)", f"{max(response_times):.4f}",
                      f"{max(response_times) * 1000:.2f} ms")
        table.add_row("Minimum Response Time (seconds)", f"{min(response_times):.4f}",
                      f"{min(response_times) * 1000:.2f} ms")
        table.add_row("Total Number of Requests", f"{len(response_times)}", "")
        table.add_row("Average Memory Usage (bytes)", f"{statistics.mean(memory_usages):.2f}",
                      f"{statistics.mean(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Median Memory Usage (bytes)", f"{statistics.median(memory_usages):.2f}",
                      f"{statistics.median(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Maximum Memory Usage (bytes)", f"{max(memory_usages):.2f}",
                      f"{max(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Minimum Memory Usage (bytes)", f"{min(memory_usages):.2f}",
                      f"{min(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Average Input Tokens", f"{statistics.mean(input_tokens):.2f}",
                      f"{statistics.mean(input_tokens):.2f} tokens/s")
        table.add_row("Average Output Tokens", f"{statistics.mean(output_tokens):.2f}",
                      f"{statistics.mean(output_tokens):.2f} tokens/s")
    else:
        table.add_row("Total Duration (seconds)", f"{duration:.2f}", f"{duration * 1000:.2f} ms")
        table.add_row("Error", "No valid response times recorded", "")

    console.print(table)


# Print system metrics
def print_system_metrics():
    cpu_usage, memory_usage, gpu_usage, gpu_memory_usage = get_system_metrics()
    table = Table(title="System Metrics")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("CPU Usage (%)", f"{cpu_usage:.2f}")
    table.add_row("Memory Usage (MB)", f"{memory_usage:.2f}")
    table.add_row("GPU Usage (%)", f"{gpu_usage:.2f}")
    table.add_row("GPU Memory Usage (MB)", f"{gpu_memory_usage:.2f}")

    console.print(table)


print_summary(
    single_thread_response_times,
    single_thread_memory_usages,
    multi_thread_input_tokens,
    multi_thread_output_tokens,
    single_thread_duration,
    "Single-threaded"
)
print_summary(
    multi_thread_response_times,
    multi_thread_memory_usages,
    multi_thread_input_tokens,
    multi_thread_output_tokens,
    multi_thread_duration,
    "Multi-threaded"
)

# Shutdown NVML
nvmlShutdown()
