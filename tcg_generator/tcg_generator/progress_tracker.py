import time
from typing import Any

def track_progress(dataset: list, entry_operation: Any):
    dataset_entries = len(dataset)
    total_time_elapsed = 0
    entry_times = list()
    for [i, entry] in enumerate(dataset):
        formatted_time = f"{time.strftime('%H:%M:%S', time.gmtime(total_time_elapsed))}"
        average_time_num = (sum(entry_times) / len(entry_times)) if len(entry_times) > 0 else 0
        average_time = f"{time.strftime('%H:%M:%S', time.gmtime(average_time_num))}"
        remaining_time = f"{time.strftime('%H:%M:%S', time.gmtime((dataset_entries - i) * average_time_num))}"
        print(f"\r{format(round(((i)/dataset_entries) * 100.0, 2), '.2f')}% - {i + 1}/{dataset_entries} - {average_time}/iteration - {formatted_time} total - {remaining_time} remaining", end="")
        start = time.time()

        entry_operation(entry)

        end = time.time()
        time_elapsed = end - start
        entry_times.append(time_elapsed)
        total_time_elapsed += time_elapsed

    formatted_time = f"{time.strftime('%H:%M:%S', time.gmtime(total_time_elapsed))}"
    average_time = f"{time.strftime('%H:%M:%S', time.gmtime((sum(entry_times) / len(entry_times)) if len(entry_times) > 0 else 0))}"
    print(f"\r{format(round(((i + 1)/dataset_entries) * 100.0, 2), '.2f')}% - {i + 1}/{dataset_entries} - {average_time}/iteration - {formatted_time} total                              ", end="")
