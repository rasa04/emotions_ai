import time
import statistics
import random
import string
import psutil
from concurrent.futures import ThreadPoolExecutor
from model import predict_emotions
from rich.console import Console
from rich.table import Table

# Initialize rich console
console = Console()


# Function to generate random text
def generate_random_text(length=100):
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + " ", k=length))


# Function to measure the response time and memory usage of the model
def measure_response_time_and_memory(text: str):
    process = psutil.Process()
    start_time = time.time()
    initial_memory = process.memory_info().rss
    predict_emotions(text)
    response_time = time.time() - start_time
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    return response_time, memory_used


# Single-threaded load test
console.print("Running single-threaded load test...", style="bold blue")
single_thread_response_times = []
single_thread_memory_usages = []
start_time = time.time()
for _ in range(100):
    text = generate_random_text()  # Generate random text
    response_time, memory_used = measure_response_time_and_memory(text)
    single_thread_response_times.append(response_time)
    single_thread_memory_usages.append(memory_used)
end_time = time.time()
single_thread_duration = end_time - start_time

# Multithreaded load test
console.print("Running multi-threaded load test...", style="bold blue")
multi_thread_response_times = []
multi_thread_memory_usages = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(measure_response_time_and_memory, generate_random_text()) for _ in range(100)]
    for future in futures:
        response_time, memory_used = future.result()
        multi_thread_response_times.append(response_time)
        multi_thread_memory_usages.append(memory_used)
end_time = time.time()
multi_thread_duration = end_time - start_time


# Print summary statistics
def print_summary(response_times, memory_usages, duration, label):
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
                      f"{min(response_times) * 1000:.4f} ms")
        table.add_row("Total Number of Requests", f"{len(response_times)}", "")
        table.add_row("Average Memory Usage (bytes)", f"{statistics.mean(memory_usages):.2f}",
                      f"{statistics.mean(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Median Memory Usage (bytes)", f"{statistics.median(memory_usages):.2f}",
                      f"{statistics.median(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Maximum Memory Usage (bytes)", f"{max(memory_usages):.2f}",
                      f"{max(memory_usages) / (1024 ** 2):.2f} MB")
        table.add_row("Minimum Memory Usage (bytes)", f"{min(memory_usages):.2f}",
                      f"{min(memory_usages) / (1024 ** 2):.2f} MB")
    else:
        table.add_row("Total Duration (seconds)", f"{duration:.2f}", f"{duration * 1000:.2f} ms")
        table.add_row("Error", "No valid response times recorded", "")

    console.print(table)


print_summary(single_thread_response_times, single_thread_memory_usages, single_thread_duration, "Single-threaded")
print_summary(multi_thread_response_times, multi_thread_memory_usages, multi_thread_duration, "Multi-threaded")
