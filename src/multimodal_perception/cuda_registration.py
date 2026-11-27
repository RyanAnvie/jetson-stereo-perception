"""Reusable CUDA depth projection with the CPU reference's visibility rules.

Geometry uses double precision, floor rasterization, nearest thermal Z, then
lowest source index for equal-Z ties. No depth subsampling or image alignment.
CUDA/NVRTC are imported only when explicitly selecting this backend.
"""
import ctypes
import hashlib

import numpy as np

from multimodal_perception.tensorrt_inference import cuda_check


SOURCE = r'''
extern "C" __global__ void clear_map(unsigned long long *zmin, int *winner, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { zmin[i]=0x7ff0000000000000ULL; winner[i]=2147483647; }
}
extern "C" __global__ void project_depth(const float *depth, const double *p,
        int *keys, double *zout, unsigned long long *zmin, int w, int h, int tw, int th) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    keys[i]=-1;
    double z=depth[i];
    if(!isfinite(z) || z<=0) return;
    double x=((i%w)-p[2])*z/p[0], y=((i/w)-p[3])*z/p[1];
    double X=p[8]*x+p[9]*y+p[10]*z+p[17];
    double Y=p[11]*x+p[12]*y+p[13]*z+p[18];
    double Z=p[14]*x+p[15]*y+p[16]*z+p[19];
    if(!isfinite(X) || !isfinite(Y) || !isfinite(Z) || Z<=0) return;
    double u=p[4]*X/Z+p[6], v=p[5]*Y/Z+p[7];
    if(!(u>=0 && u<tw && v>=0 && v<th)) return;
    int key=(int)floor(v)*tw+(int)floor(u);
    keys[i]=key; zout[i]=Z;
    atomicMin(zmin+key,(unsigned long long)__double_as_longlong(Z));
}
extern "C" __global__ void choose_source(const int *keys, const double *zout,
        const unsigned long long *zmin, int *winner, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && keys[i]>=0 && (unsigned long long)__double_as_longlong(zout[i])==zmin[keys[i]])
        atomicMin(winner+keys[i],i);
}
extern "C" __global__ void gather_visible(const float *depth, const unsigned char *rgb,
        const double *p, const int *winner, double *points, unsigned char *image,
        unsigned char *observed, int w, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int source=winner[i];
    observed[i]=source!=2147483647 ? 255 : 0;
    for(int c=0;c<3;c++) image[i*3+c]=source!=2147483647 ? rgb[source*3+c] : 114;
    if(source==2147483647) return;
    double z=depth[source];
    points[i*3]=((source%w)-p[2])*z/p[0];
    points[i*3+1]=((source/w)-p[3])*z/p[1];
    points[i*3+2]=z;
}
extern "C" __global__ void pack_inputs(const unsigned char *rgb, const unsigned char *thermal,
                                      float *rgb_tensor, float *thermal_tensor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=640*640) return;
    int y=i/640, x=i%640;
    bool content=y>=64 && y<576;
    int j=(y-64)*640+x;
    float heat=__fdiv_rn(float(content ? thermal[j] : 114),255.0f);
    for(int c=0;c<3;c++) {
        rgb_tensor[c*640*640+i]=__fdiv_rn(float(content ? rgb[j*3+2-c] : 114),255.0f);
        thermal_tensor[c*640*640+i]=heat;
    }
}
extern "C" __global__ void scan_visible(const int *winner, int *prefix, int *counts, int n) {
    __shared__ int scan[256];
    int lane=threadIdx.x, i=blockIdx.x*256+lane;
    int present=i<n && winner[i]!=2147483647;
    scan[lane]=present;
    __syncthreads();
    for(int step=1;step<256;step*=2) {
        int other=lane>=step ? scan[lane-step] : 0;
        __syncthreads();
        scan[lane]+=other;
        __syncthreads();
    }
    if(i<n) prefix[i]=scan[lane]-present;
    if(lane==255) counts[blockIdx.x]=scan[lane];
}
extern "C" __global__ void scan_blocks(int *counts, int *total, int n) {
    if(blockIdx.x || threadIdx.x) return;
    int offset=0;
    for(int i=0;i<n;i++) {int count=counts[i];counts[i]=offset;offset+=count;}
    *total=offset;
}
// Only distances <=2 are filled. The three distance groups and tie priorities
// reproduce OpenCV's forward/backward 5x5 chamfer-label traversal at this radius.
__device__ int hole_source(const int *winner, int x, int y, int w, int h) {
    const int dx[12]={0,-1,0,1, -1,1,1,-1, 0,-2,0,2};
    const int dy[12]={-1,0,1,0, -1,-1,1,1, -2,0,2,0};
    for(int j=0;j<12;j++) {
        int xx=x+dx[j], yy=y+dy[j];
        if(xx>=0 && xx<w && yy>=0 && yy<h) {
            int source=winner[yy*w+xx];
            if(source!=2147483647) return source;
        }
    }
    return 2147483647;
}
extern "C" __global__ void export_visible(const float *depth, const unsigned char *rgb,
        const unsigned char *thermal, const double *p, const int *winner, const int *prefix,
        const int *offsets, double *points, int *rgb_pixels, int *thermal_pixels,
        unsigned char *intensity, unsigned char *image, unsigned char *observed,
        unsigned char *usable, int w, int tw, int th) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tw*th) return;
    int source=winner[i], x=i%tw, y=i/tw;
    bool visible=source!=2147483647;
    observed[i]=visible ? 255 : 0;
    if(visible) {
        int j=offsets[i/256]+prefix[i];
        double z=depth[source];
        points[j*3]=((source%w)-p[2])*z/p[0];
        points[j*3+1]=((source/w)-p[3])*z/p[1];
        points[j*3+2]=z;
        rgb_pixels[j*2]=source%w;rgb_pixels[j*2+1]=source/w;
        thermal_pixels[j*2]=x;thermal_pixels[j*2+1]=y;
        intensity[j]=thermal[i];
    } else source=hole_source(winner,x,y,tw,th);
    usable[i]=source!=2147483647 ? 255 : 0;
    for(int c=0;c<3;c++) image[i*3+c]=source!=2147483647 ? rgb[source*3+c] : 114;
}
extern "C" __global__ void mark_predictions(const float *raw, int *selected, int *invalid,
                                          float threshold, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const float *row=raw+i*11;
    selected[i]=2147483647;
    for(int c=0;c<11;c++) if(!isfinite(row[c])) {atomicOr(invalid,1);return;}
    float best=row[4]*row[5];
    for(int c=6;c<11;c++) {float score=row[4]*row[c];if(score>best) best=score;}
    if(best>=threshold) selected[i]=i;
}
extern "C" __global__ void export_predictions(const float *raw, const int *selected,
        const int *prefix, const int *offsets, float *rows, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || selected[i]==2147483647) return;
    int j=offsets[i/256]+prefix[i];
    for(int c=0;c<11;c++) rows[j*11+c]=raw[i*11+c];
}
'''


class CudaRegistration:
    def __init__(self, rgb_shape=(540, 960), thermal_shape=(512, 640), compact=False, mapped_outputs=False):
        from cuda import cuda, cudart, nvrtc
        self.driver, self.cuda = cuda, cudart
        self.buffers, self.module, self.stream = {}, None, None
        self.closed = False
        self.compact = compact
        self.mapped_outputs = mapped_outputs
        self.launch_cache = {}
        self.rgb_shape, self.thermal_shape = tuple(rgb_shape), tuple(thermal_shape)
        try:
            # Retain the runtime primary context shared with TensorRT.
            cuda_check(cudart.cudaFree(0))
            device = cuda_check(cudart.cudaGetDevice())
            prop = cuda_check(cudart.cudaGetDeviceProperties(device))
            if mapped_outputs and (not compact or not prop.integrated or not prop.canMapHostMemory or (prop.major, prop.minor) < (7, 2)):
                raise ValueError("Mapped output mode requires compact geometry and an I/O-coherent integrated GPU")
            program = cuda_check(nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"registration.cu", 0, [], []))
            try:
                opts = [b"--fmad=false", f"--gpu-architecture=compute_{prop.major}{prop.minor}".encode()]
                result = nvrtc.nvrtcCompileProgram(program, len(opts), opts)
                if int(result[0]):
                    log = b" "*cuda_check(nvrtc.nvrtcGetProgramLogSize(program))
                    cuda_check(nvrtc.nvrtcGetProgramLog(program, log))
                    raise RuntimeError(log.decode())
                ptx = b" "*cuda_check(nvrtc.nvrtcGetPTXSize(program))
                cuda_check(nvrtc.nvrtcGetPTX(program, ptx))
                self.module = cuda_check(cuda.cuModuleLoadData(ptx))
            finally:
                cuda_check(nvrtc.nvrtcDestroyProgram(program))
            self.kernels = {name: cuda_check(cuda.cuModuleGetFunction(self.module, name.encode()))
                            for name in ["clear_map", "project_depth", "choose_source", "gather_visible", "pack_inputs",
                                         "scan_visible", "scan_blocks", "export_visible", "mark_predictions", "export_predictions"]}
            self.stream = cuda_check(cudart.cudaStreamCreate())
            h, w = self.rgb_shape
            th, tw = self.thermal_shape
            for name, shape, dtype in [
                ("depth", (h, w), np.float32), ("rgb", (h, w, 3), np.uint8),
                ("params", (20,), np.float64), ("keys", (h*w,), np.int32),
                ("zout", (h*w,), np.float64), ("zmin", (th*tw,), np.uint64),
                ("winner", (th*tw,), np.int32), ("points", (th*tw, 3), np.float64),
                ("image", (th, tw, 3), np.uint8), ("observed", (th, tw), np.uint8),
                ("thermal", (th, tw), np.uint8)]:
                self._allocate(name, shape, dtype, name not in {"keys", "zout", "zmin"})
            if compact:
                for name, shape, dtype, host in [
                    ("prefix", (th*tw,), np.int32, False), ("offsets", ((th*tw+255)//256,), np.int32, False),
                    ("total", (1,), np.int32, True), ("rgb_pixels", (th*tw, 2), np.int32, True),
                    ("thermal_pixels", (th*tw, 2), np.int32, True), ("intensity", (th*tw,), np.uint8, True),
                    ("usable", (th, tw), np.uint8, True),
                    ("prediction_rows", (102000, 11), np.float32, True), ("prediction_invalid", (1,), np.int32, True)]:
                    self._allocate(name, shape, dtype, host)
            self.metadata = {"backend": "CUDA NVRTC", "geometry_precision": "float64_no_fma",
                             "gpu_compaction_and_hole_fill": compact,
                             "mapped_output_buffers": mapped_outputs,
                             "kernel_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                             "nvrtc_version": list(cuda_check(nvrtc.nvrtcVersion())),
                             "depth_stride": 1, "persistent_device_bytes": sum(b["nbytes"] for b in self.buffers.values() if not b["mapped"]),
                             "persistent_pinned_bytes": sum(b["nbytes"] for b in self.buffers.values() if "host" in b),
                             "mapped_shared_bytes": sum(b["nbytes"] for b in self.buffers.values() if b["mapped"]),
                             "visibility": "nearest thermal Z; equal-Z ties use first RGB source pixel"}
        except BaseException:
            self.close()
            raise

    def _allocate(self, name, shape, dtype, host):
        nbytes = int(np.prod(shape))*np.dtype(dtype).itemsize
        mapped = self.mapped_outputs and host and name in {"points", "rgb_pixels", "thermal_pixels", "intensity", "observed", "usable", "total", "prediction_rows", "prediction_invalid"}
        b = {"device": None, "host_pointer": None, "nbytes": nbytes, "mapped": mapped}
        self.buffers[name] = b
        if mapped:
            b["host_pointer"] = cuda_check(self.cuda.cudaHostAlloc(nbytes, self.cuda.cudaHostAllocMapped))
            b["device"] = cuda_check(self.cuda.cudaHostGetDevicePointer(b["host_pointer"], 0))
        else:
            b["device"] = cuda_check(self.cuda.cudaMalloc(nbytes))
        if host:
            if not mapped:
                b["host_pointer"] = cuda_check(self.cuda.cudaMallocHost(nbytes))
            storage = (ctypes.c_ubyte*nbytes).from_address(int(b["host_pointer"]))
            b["host"] = np.ctypeslib.as_array(storage).view(dtype).reshape(shape)

    def _launch(self, name, n, pointers, integers, stream=None):
        key = (name, tuple(pointers), tuple(integers))
        if key not in self.launch_cache:
            values = [np.asarray([int(self.buffers[key]["device"])], dtype=np.uint64) for key in pointers]
            values += [np.asarray([value], dtype=np.int32) for value in integers]
            args = np.asarray([v.ctypes.data for v in values], dtype=np.uint64)
            self.launch_cache[key] = (values, args)
        args = self.launch_cache[key][1]
        cuda_check(self.driver.cuLaunchKernel(self.kernels[name], (n+255)//256, 1, 1,
                                             256, 1, 1, 0, int(self.stream if stream is None else stream), args.ctypes.data, 0))

    def fuse(self, sample, calibration, max_time_error_s=.032):
        if self.closed:
            raise RuntimeError("CUDA registration is closed")
        times = np.array([sample["timestamp_s"], sample["rgb_timestamp_s"], sample["thermal_timestamp_s"]])
        if (not sample["valid"] or sample["ffc"] or not np.isfinite(times).all()
                or np.max(np.abs(times[1:]-times[0])) > max_time_error_s):
            raise ValueError("Unsynchronized modalities must not enter CUDA registration")
        h, w = self.rgb_shape
        th, tw = self.thermal_shape
        if (sample["depth"].shape != (h, w) or sample["depth"].dtype != np.float32
                or sample["rgb"].shape != (h, w, 3) or sample["rgb"].dtype != np.uint8
                or sample["thermal"].shape != (th, tw) or sample["thermal"].dtype != np.uint8
                or (calibration["rgb"]["height"], calibration["rgb"]["width"]) != (h, w)
                or (calibration["thermal"]["height"], calibration["thermal"]["width"]) != (th, tw)):
            raise ValueError("CUDA registration input domain mismatch")
        fixed = calibration["transform"]
        if fixed["source"] != calibration["rgb"]["frame"] or fixed["target"] != calibration["thermal"]["frame"]:
            raise ValueError("CUDA registration optical frame mismatch")
        kr, kt = calibration["rgb"]["K"], calibration["thermal"]["K"]
        params = np.r_[kr[0, 0], kr[1, 1], kr[0, 2], kr[1, 2], kt[0, 0], kt[1, 1], kt[0, 2], kt[1, 2],
                       fixed["R"].ravel(), fixed["t"]]
        inputs = [("depth", sample["depth"]), ("rgb", sample["rgb"]), ("params", params)]
        if self.compact:
            inputs.append(("thermal", sample["thermal"]))
        for name, value in inputs:
            b = self.buffers[name]
            np.copyto(b["host"], value)
            cuda_check(self.cuda.cudaMemcpyAsync(b["device"], b["host_pointer"], b["nbytes"],
                                                 self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        self._launch("clear_map", th*tw, ["zmin", "winner"], [th*tw])
        self._launch("project_depth", h*w, ["depth", "params", "keys", "zout", "zmin"], [w, h, tw, th])
        self._launch("choose_source", h*w, ["keys", "zout", "zmin", "winner"], [h*w])
        if self.compact:
            self._launch("scan_visible", th*tw, ["winner", "prefix", "offsets"], [th*tw])
            self._launch("scan_blocks", 1, ["offsets", "total"], [(th*tw+255)//256])
            self._launch("export_visible", th*tw, ["depth", "rgb", "thermal", "params", "winner", "prefix", "offsets",
                                                  "points", "rgb_pixels", "thermal_pixels", "intensity", "image", "observed", "usable"], [w, tw, th])
            for name in ["total", "points", "rgb_pixels", "thermal_pixels", "intensity", "image", "observed", "usable"]:
                b = self.buffers[name]
                if b["mapped"]:
                    continue
                cuda_check(self.cuda.cudaMemcpyAsync(b["host_pointer"], b["device"], b["nbytes"],
                                                     self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
            cuda_check(self.cuda.cudaStreamSynchronize(self.stream))
            count = int(self.buffers["total"]["host"][0])
            fusion = {name: self.buffers[key]["host"][:count].copy() for name, key in
                      [("points", "points"), ("rgb_pixels", "rgb_pixels"), ("thermal_pixels", "thermal_pixels"), ("intensity_u8", "intensity")]}
            fusion.update(sampled_depth_points=int(np.count_nonzero(np.isfinite(sample["depth"]) & (sample["depth"] > 0))),
                          status=calibration.get("evidence", {}).get("status", "source_reconstructed_CAD_nominal"),
                          gpu_usable_mask=self.buffers["usable"]["host"].copy())
            return fusion, self.buffers["image"]["host"].copy(), self.buffers["observed"]["host"].copy()
        self._launch("gather_visible", th*tw, ["depth", "rgb", "params", "winner", "points", "image", "observed"], [w, th*tw])
        for name in ["winner", "points", "image", "observed"]:
            b = self.buffers[name]
            cuda_check(self.cuda.cudaMemcpyAsync(b["host_pointer"], b["device"], b["nbytes"],
                                                 self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        cuda_check(self.cuda.cudaStreamSynchronize(self.stream))
        winner = self.buffers["winner"]["host"]
        occupied = np.flatnonzero(winner != np.iinfo(np.int32).max)
        sources = winner[occupied]
        uv = np.column_stack((occupied % tw, occupied // tw))
        fusion = {"points": self.buffers["points"]["host"][occupied],
                  "rgb_pixels": np.column_stack((sources % w, sources // w)), "thermal_pixels": uv,
                  "intensity_u8": sample["thermal"].ravel()[occupied],
                  "sampled_depth_points": int(np.count_nonzero(np.isfinite(sample["depth"]) & (sample["depth"] > 0))),
                  "status": calibration.get("evidence", {}).get("status", "source_reconstructed_CAD_nominal")}
        # Returned images own their memory; a later frame cannot overwrite them.
        return fusion, self.buffers["image"]["host"].copy(), self.buffers["observed"]["host"].copy()

    def prepare_engine_inputs(self, rgb, thermal, engine, resident=False):
        """Pack bytes directly into existing TensorRT device buffers on its stream.

        No float image copies, per-frame allocations, resize, or precision change.
        The caller must enqueue inference on this same stream before reusing input.
        """
        if self.closed or engine.get("closed"):
            raise RuntimeError("CUDA registration or TensorRT engine is closed")
        if rgb.shape != (512, 640, 3) or rgb.dtype != np.uint8 or thermal.shape != (512, 640) or thermal.dtype != np.uint8:
            raise ValueError("CUDA preprocessing expects the audited image domain")
        if set(engine["input_names"]) != {"rgb", "thermal"}:
            raise ValueError("CUDA preprocessing requires both joint input names")
        for name in ["rgb", "thermal"]:
            host = engine["buffers"][name]["host"]
            if host.shape != (1, 3, 640, 640) or host.dtype != np.float32:
                raise ValueError("CUDA preprocessing requires FP32 I/O tensors")
        stream = engine["stream"]
        if resident and not self.compact:
            raise ValueError("Resident preprocessing requires compact GPU registration")
        for name, value in ([] if resident else [("image", rgb), ("thermal", thermal)]):
            b = self.buffers[name]
            np.copyto(b["host"], value)
            cuda_check(self.cuda.cudaMemcpyAsync(b["device"], b["host_pointer"], b["nbytes"],
                                                 self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        ptrs = [self.buffers["image"]["device"], self.buffers["thermal"]["device"],
                engine["buffers"]["rgb"]["device"], engine["buffers"]["thermal"]["device"]]
        values = [np.asarray([int(p)], dtype=np.uint64) for p in ptrs]
        args = np.asarray([v.ctypes.data for v in values], dtype=np.uint64)
        cuda_check(self.driver.cuLaunchKernel(self.kernels["pack_inputs"], 1600, 1, 1, 256, 1, 1, 0,
                                             int(stream), args.ctypes.data, 0))

    def close(self):
        if self.closed:
            return
        if self.stream is not None:
            cuda_check(self.cuda.cudaStreamSynchronize(self.stream))
        for b in self.buffers.values():
            if b["device"] is not None and not b["mapped"]:
                cuda_check(self.cuda.cudaFree(b["device"]))
            if b["host_pointer"] is not None:
                cuda_check(self.cuda.cudaFreeHost(b["host_pointer"]))
        self.buffers.clear()
        if self.stream is not None:
            cuda_check(self.cuda.cudaStreamDestroy(self.stream))
        if self.module is not None:
            cuda_check(self.driver.cuModuleUnload(self.module))
        self.closed = True

    def filter_predictions(self, engine, threshold):
        """Stable GPU candidate filtering; reuses completed geometry scan scratch.

        No top-K limit, score approximation, or NMS change. All raw values are
        checked for finiteness, including rows below the detection threshold.
        """
        if not self.compact or np.prod(self.rgb_shape) < 102000 or np.prod(self.thermal_shape) < 102000:
            raise ValueError("GPU candidate filtering requires full-size compact buffers")
        raw = engine["buffers"]["predictions"]
        if raw["host"].shape != (1, 102000, 11) or raw["host"].dtype != np.float32:
            raise ValueError("Unexpected candidate filter input contract")
        stream = engine["stream"]
        cuda_check(self.cuda.cudaMemsetAsync(self.buffers["prediction_invalid"]["device"], 0, 4, stream))
        key = ("candidate_kernels", int(raw["device"]), float(threshold))
        if key not in self.launch_cache:
            def args_for(pointers, tail):
                values = [np.asarray([int(p)], np.uint64) for p in pointers]+tail
                return values, np.asarray([v.ctypes.data for v in values], np.uint64)
            mark = args_for([raw["device"], self.buffers["keys"]["device"], self.buffers["prediction_invalid"]["device"]],
                            [np.asarray([threshold], np.float32), np.asarray([102000], np.int32)])
            export = args_for([raw["device"], self.buffers["keys"]["device"], self.buffers["prefix"]["device"],
                               self.buffers["offsets"]["device"], self.buffers["prediction_rows"]["device"]],
                              [np.asarray([102000], np.int32)])
            self.launch_cache[key] = (mark, export)
        mark, export = self.launch_cache[key]
        cuda_check(self.driver.cuLaunchKernel(self.kernels["mark_predictions"], 399, 1, 1, 256, 1, 1, 0,
                                             int(stream), mark[1].ctypes.data, 0))
        self._launch("scan_visible", 102000, ["keys", "prefix", "offsets"], [102000], stream)
        self._launch("scan_blocks", 1, ["offsets", "total"], [399], stream)
        cuda_check(self.driver.cuLaunchKernel(self.kernels["export_predictions"], 399, 1, 1, 256, 1, 1, 0,
                                             int(stream), export[1].ctypes.data, 0))
        for name in ["total", "prediction_invalid", "prediction_rows"]:
            b = self.buffers[name]
            if b["mapped"]:
                continue
            cuda_check(self.cuda.cudaMemcpyAsync(b["host_pointer"], b["device"], b["nbytes"],
                                                 self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        # Adapter records its final event and synchronizes before reading these.
        return self.read_filtered_predictions

    def read_filtered_predictions(self):
        if self.buffers["prediction_invalid"]["host"][0]:
            raise ValueError("Invalid joint detector output")
        count = int(self.buffers["total"]["host"][0])
        return {"predictions": self.buffers["prediction_rows"]["host"][:count].copy()[None]}
