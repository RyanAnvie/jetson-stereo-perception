"""Bounded decoded-frame reuse and RGB motion preparation ahead of CUDA work."""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from threading import Lock
import time

from multimodal_perception.dataset import load_frame
from multimodal_perception.rig_motion import load_exposure_sample, calibration_at_exposure


def prepared_samples(dataset, indices, calibration, temporal, ahead=2, workers=1):
    if not 1 <= ahead <= 8:
        raise ValueError("Preparation queue must contain 1 to 8 samples")
    cache = OrderedDict()
    lock = Lock()
    if not 1 <= workers <= min(4, ahead):
        raise ValueError("Preparation workers must be 1 to 4 and no more than queue depth")

    def cached_frame(dataset, index):
        with lock:
            owner = index not in cache
            if owner:
                cache[index] = Future()
            future = cache[index]
            cache.move_to_end(index)
            # Do not evict in-flight decodes; a neighbor can share its Future.
            for key in list(cache):
                if len(cache) <= ahead+4:
                    break
                if cache[key].done():
                    del cache[key]
        if owner:
            try:
                future.set_result(load_frame(dataset, index))
            except BaseException as exc:
                future.set_exception(exc)
        return future.result()

    def prepare(index, submitted):
        started = time.perf_counter()
        sample = load_exposure_sample(dataset, index, calibration, temporal, loader=cached_frame)
        loaded = time.perf_counter()
        sample["prepared_exposure"] = calibration_at_exposure(sample, calibration, temporal)
        sample["preparation_timing"] = {"io_decode_ms": (loaded-started)*1000,
                                        "rig_motion_ms": (time.perf_counter()-loaded)*1000}
        sample["preparation_started"] = started
        sample["preparation_submitted"] = submitted
        sample["preparation_finished"] = time.perf_counter()
        return sample

    # OpenCV's RNG is thread-local and each motion call reseeds it. Results are
    # delivered in source order, regardless of decode/estimation completion order.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rgb-motion") as executor:
        pending = deque()
        iterator = iter(indices)
        try:
            for _ in range(ahead):
                index = next(iterator, None)
                if index is not None:
                    pending.append(executor.submit(prepare, index, time.perf_counter()))
            while pending:
                sample = pending.popleft().result()
                index = next(iterator, None)
                if index is not None:
                    pending.append(executor.submit(prepare, index, time.perf_counter()))
                yield sample
        finally:
            for future in pending:
                future.cancel()
