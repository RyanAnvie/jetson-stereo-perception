from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest


def fake_gpu(monkeypatch):
    import multimodal_perception.joint_workers as module
    monkeypatch.setattr(module, "activate_cuda", lambda: None)
    monkeypatch.setattr(module, "process_joint_sample", lambda *args: ([], {}, {"processing_ms": 0.}))
    return module


def test_completed_output_is_delivered_while_next_input_is_blocked(monkeypatch):
    module = fake_gpu(monkeypatch)
    release, closed = Event(), Event()
    pulled = []
    def source():
        try:
            for index in range(5):
                pulled.append(index)
                if index == 2:
                    assert release.wait(5)
                yield {"index": index}
        finally:
            closed.set()
    stream = module.processed_samples(source(), [{"backend": "tensorrt"} for _ in range(2)], {}, {"inference_dispatch": "independent"})
    with ThreadPoolExecutor(max_workers=1) as consumer:
        try:
            assert consumer.submit(next, stream).result(timeout=2)[0]["index"] == 0
            assert consumer.submit(next, stream).result(timeout=2)[0]["index"] == 1
            assert len(pulled) <= 3  # Two GPU slots plus one source pull; no runaway prefetch.
        finally:
            release.set()
            stream.close()
    assert closed.is_set()


@pytest.mark.parametrize("failure", ["source", "gpu"])
def test_independent_dispatch_propagates_failures_and_closes_source(monkeypatch, failure):
    module = fake_gpu(monkeypatch)
    closed = Event()
    if failure == "gpu":
        def fail(*args):
            raise RuntimeError("gpu failed")
        monkeypatch.setattr(module, "process_joint_sample", fail)
    def source():
        try:
            yield {"index": 0}
            if failure == "source":
                raise RuntimeError("source failed")
            yield {"index": 1}
        finally:
            closed.set()
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        list(module.processed_samples(source(), [{"backend": "tensorrt"} for _ in range(2)], {}, {"inference_dispatch": "independent"}))
    assert closed.is_set()
