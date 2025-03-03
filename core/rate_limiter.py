import time
import logging
from collections import deque

from core.singleton import singleton


@singleton
class RateLimiter:
    def __init__(self, max_calls: int=60, period: int=60):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = deque()

    def call(self, func, *args, **kwargs):
        current_time = time.time()
        while self.timestamps and self.timestamps[0] <= current_time - self.period:
            self.timestamps.popleft()

        if len(self.timestamps) < self.max_calls:
            self.timestamps.append(current_time)
            return func(*args, **kwargs)
        else:
            wait_time = self.period - (current_time - self.timestamps[0])
            logging.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            self.timestamps.append(time.time())
            return func(*args, **kwargs)
