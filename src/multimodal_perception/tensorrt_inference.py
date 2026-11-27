"""Synchronous, single-stream adapter for audited static TensorRT 10 engines."""

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np


def cuda_check(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def load_engine(path, model_sha256, contract=None):
    import tensorrt as trt
    from cuda import cudart

    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This adapter is validated for TensorRT 10 only")
    path = Path(path)
    content = path.read_bytes()
    provenance = json.loads(path.with_suffix(".json").read_text())
    digest = hashlib.sha256(content).hexdigest()
    if provenance["engine_sha256"] != digest or provenance["model_sha256"] != model_sha256:
        raise ValueError("Engine/model checksum differs from build provenance")
    if provenance["tensorrt"] != trt.__version__:
        raise ValueError("Rebuild the engine with the installed TensorRT version")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(content)
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an execution context")
    state = {"cuda": cudart, "logger": logger, "runtime": runtime, "engine": engine,
             "context": context, "stream": None, "buffers": {}, "events": []}
    try:
        state["stream"] = cuda_check(cudart.cudaStreamCreate())
        io = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if (not shape or any(d <= 0 for d in shape) or dtype not in [np.dtype("float32"), np.dtype("float16")]
                    or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError(f"Unsupported engine I/O: {name}, {shape}, {dtype}")
            nbytes = int(np.prod(shape)) * dtype.itemsize
            buffer = {"device": None, "host_pointer": None}
            state["buffers"][name] = buffer
            buffer["device"] = cuda_check(cudart.cudaMalloc(nbytes))
            buffer["host_pointer"] = cuda_check(cudart.cudaMallocHost(nbytes))
            storage = (ctypes.c_ubyte * nbytes).from_address(int(buffer["host_pointer"]))
            buffer["host"] = np.ctypeslib.as_array(storage).view(dtype).reshape(shape)
            if not context.set_tensor_address(name, int(buffer["device"])):
                raise RuntimeError(f"Could not bind TensorRT tensor {name}")
            io.append({"name": name, "shape": list(shape), "dtype": str(dtype),
                       "mode": "input" if is_input else "output", "format": "LINEAR"})
        inputs = [item for item in io if item["mode"] == "input"]
        outputs = [item for item in io if item["mode"] == "output"]
        if contract is None:
            if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, 416, 416]:
                raise ValueError(f"Expected one static YOLOX input: {inputs}")
            if len(outputs) != 1 or outputs[0]["shape"] != [1, 3549, 85]:
                raise ValueError(f"Expected one raw YOLOX output: {outputs}")
        elif ({item["name"]: item["shape"] for item in inputs} != contract["inputs"]
              or {item["name"]: item["shape"] for item in outputs} != contract["outputs"]):
            raise ValueError("Engine I/O differs from the explicit model contract")
        state["input_names"] = [item["name"] for item in inputs]
        state["output_names"] = [item["name"] for item in outputs]
        state["input_name"], state["output_name"] = inputs[0]["name"], outputs[0]["name"]
        for _ in range(4):
            state["events"].append(cuda_check(cudart.cudaEventCreate()))
        state["metadata"] = {"backend": "tensorrt", "provider": "TensorRT CUDA",
                             "runtime": trt.__version__, "engine_sha256": digest,
                             "model_sha256": model_sha256, "precision": provenance["precision"],
                             "cuda_runtime": cuda_check(cudart.cudaRuntimeGetVersion()),
                             "cuda_driver": cuda_check(cudart.cudaDriverGetVersion()), "io": io,
                             "build": provenance, "timing": "CUDA events on one stream; synchronized before returning"}
        return state
    except BaseException:
        close_engine(state)
        raise


def infer_engine(state, tensor):
    if state.get("closed"):
        raise RuntimeError("TensorRT engine is closed")
    outputs, timing = infer_engine_multi(state, {state["input_name"]: tensor})
    return outputs[state["output_name"]], timing


def infer_engine_multi(state, tensors):
    if state.get("closed"):
        raise RuntimeError("TensorRT engine is closed")
    if set(tensors) != set(state["input_names"]):
        raise ValueError("Input tensor names do not match the engine")
    for name, tensor in tensors.items():
        source = state["buffers"][name]
        if tensor.shape != source["host"].shape or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid input tensor: {name}")
        np.copyto(source["host"], tensor, casting="same_kind")
    return infer_engine_bound(state, copy_inputs=True)


def infer_engine_bound(state, copy_inputs=False):
    """Execute on bound buffers; a device producer must use this state's stream."""
    if state.get("closed"):
        raise RuntimeError("TensorRT engine is closed")
    cuda, stream = state["cuda"], state["stream"]
    events = state["events"]
    cuda_check(cuda.cudaEventRecord(events[0], stream))
    if copy_inputs:
        for name in state["input_names"]:
            source = state["buffers"][name]
            cuda_check(cuda.cudaMemcpyAsync(source["device"], source["host_pointer"], source["host"].nbytes,
                                            cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    cuda_check(cuda.cudaEventRecord(events[1], stream))
    if state.get("graph_instance") is not None:
        cuda_check(cuda.cudaGraphLaunch(state["graph_instance"], stream))
    elif not state["context"].execute_async_v3(stream_handle=int(stream)):
        raise RuntimeError("TensorRT execution failed")
    cuda_check(cuda.cudaEventRecord(events[2], stream))
    read_filtered = None
    if state.get("output_filter") is not None:
        read_filtered = state["output_filter"](state)
    else:
        for name in state["output_names"]:
            target = state["buffers"][name]
            cuda_check(cuda.cudaMemcpyAsync(target["host_pointer"], target["device"], target["host"].nbytes,
                                            cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    cuda_check(cuda.cudaEventRecord(events[3], stream))
    cuda_check(cuda.cudaStreamSynchronize(stream))
    timing = {}
    output_key = "candidate_filter_d2h_ms" if read_filtered else "d2h_ms"
    for key, start, end in [("h2d_ms", 0, 1), ("model_compute_ms", 1, 2), (output_key, 2, 3)]:
        timing[key] = float(cuda_check(cuda.cudaEventElapsedTime(events[start], events[end])))
    outputs = read_filtered() if read_filtered else {name: np.array(state["buffers"][name]["host"], dtype=np.float32, copy=True)
                                                     for name in state["output_names"]}
    if state.get("cuda_graph_requested") and state.get("graph_instance") is None:
        capture_engine_graph(state)
    return outputs, timing


def capture_engine_graph(state):
    """Capture only inference after a completed warmup; shapes/addresses stay fixed."""
    cuda, stream = state["cuda"], state["stream"]
    cuda_check(cuda.cudaStreamBeginCapture(stream, cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
    try:
        success = state["context"].execute_async_v3(stream_handle=int(stream))
    except BaseException:
        ended = cuda.cudaStreamEndCapture(stream)
        if int(ended[0]) == 0:
            cuda.cudaGraphDestroy(ended[1])
        raise
    graph = cuda_check(cuda.cudaStreamEndCapture(stream))
    state["graph"] = graph
    if not success:
        raise RuntimeError("TensorRT execution failed during graph capture")
    state["graph_instance"] = cuda_check(cuda.cudaGraphInstantiate(graph, 0))
    state["metadata"]["cuda_graph"] = True


def close_engine(state):
    """Release only resources owned by this adapter; never reset the CUDA device."""
    if state.get("closed"):
        return
    cuda = state["cuda"]
    errors = []
    if state["stream"] is not None:
        errors.append(cuda.cudaStreamSynchronize(state["stream"])[0])
    if state.get("graph_instance") is not None:
        errors.append(cuda.cudaGraphExecDestroy(state.pop("graph_instance"))[0])
    if state.get("graph") is not None:
        errors.append(cuda.cudaGraphDestroy(state.pop("graph"))[0])
    state["context"] = None
    for buffer in state["buffers"].values():
        if buffer["device"] is not None:
            errors.append(cuda.cudaFree(buffer["device"])[0])
        if buffer["host_pointer"] is not None:
            errors.append(cuda.cudaFreeHost(buffer["host_pointer"])[0])
    state["buffers"].clear()
    for event in state["events"]:
        errors.append(cuda.cudaEventDestroy(event)[0])
    if state["stream"] is not None:
        errors.append(cuda.cudaStreamDestroy(state["stream"])[0])
    state["engine"] = None
    state["runtime"] = None
    state["closed"] = True
    failures = [str(error) for error in errors if int(error) != 0]
    if failures:
        raise RuntimeError(f"CUDA resource cleanup failed: {failures}")
