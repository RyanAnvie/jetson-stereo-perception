"""Bounded ordered GPU processing with an independent context per worker."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Event, Thread
import time

from multimodal_perception.joint_detection import process_joint_sample


def activate_cuda():
    from cuda import cudart
    from multimodal_perception.tensorrt_inference import cuda_check
    cuda_check(cudart.cudaSetDevice(0))


def processed_samples(source, detectors, calibration, config):
    dispatch = config.get("inference_dispatch", "inline_refill")
    if dispatch not in {"inline_refill", "independent"}:
        raise ValueError("Unknown inference dispatch mode")
    if not 1 <= len(detectors) <= 3 or (len(detectors) == 1 and dispatch != "independent"):
        raise ValueError("GPU processing requires 1 to 3 independent states; one state requires independent dispatch")
    if any(d["backend"] != "tensorrt" for d in detectors):
        raise ValueError("GPU processing workers require TensorRT")
    executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"inference-{i}", initializer=activate_cuda)
                 for i in range(len(detectors))]
    pending = deque()

    def process(sample, detector, submitted):
        started = time.perf_counter()
        sample["inference_started"] = started
        detections, fusion, timing = process_joint_sample(sample, detector, calibration, config)
        sample["inference_finished"] = time.perf_counter()
        timing["gpu_queue_wait_ms"] = (started-submitted)*1000
        return sample, detections, fusion, timing

    def submit(worker):
        sample = next(source, None)
        if sample is None:
            return False
        future = executors[worker].submit(process, sample, detectors[worker], time.perf_counter())
        pending.append((worker, future))
        return True

    if dispatch == "independent":
        # Source access has one owner. Its waits cannot hold up completed GPU
        # outputs. Worker tokens bound submitted GPU jobs to len(detectors).
        available, jobs, stopped = Queue(), Queue(), Event()
        for worker in range(len(detectors)):
            available.put(worker)

        def feed():
            error = None
            try:
                while not stopped.is_set():
                    worker = available.get()
                    if worker is None or stopped.is_set():
                        break
                    sample = next(source, None)
                    if sample is None or stopped.is_set():
                        break
                    future = executors[worker].submit(process, sample, detectors[worker], time.perf_counter())
                    jobs.put((worker, future))
            except BaseException as exc:
                error = exc
            finally:
                try:
                    source.close()
                except BaseException as exc:
                    error = error or exc
                jobs.put((None, error))

        feeder = Thread(target=feed, name="inference-dispatch")
        feeder.start()
        try:
            while True:
                worker, future = jobs.get()
                if worker is None:
                    if future is not None:
                        raise future
                    break
                result = future.result()
                available.put(worker)
                yield result
        finally:
            stopped.set()
            available.put(None)
            feeder.join()
            while True:
                try:
                    worker, future = jobs.get_nowait()
                except Empty:
                    break
                if worker is not None:
                    future.cancel()
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
        return

    try:
        for worker in range(len(detectors)):
            if not submit(worker):
                break
        while pending:
            worker, future = pending.popleft()
            result = future.result()
            # Refill this worker before handing a completed result to rendering.
            submit(worker)
            yield result
    finally:
        for _, future in pending:
            future.cancel()
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
        source.close()
