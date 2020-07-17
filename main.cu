#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <thread>

#include <cuda.h>

#include <cuda_runtime.h>

template <typename T, typename S>
T round_up (T num, S factor)
{
  return num + factor - 1 - (num - 1) % factor;
}

void chck (CUresult status)
{
  if (status != CUDA_SUCCESS)
    {
      const char *error_str {};
      cuGetErrorString (status, &error_str);
      throw std::runtime_error ("Error: " + std::string (error_str));
    }
}

size_t get_granularity (int dev)
{
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;

  size_t granularity;
  chck (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return granularity;
}

template <typename data_type>
class physical_memory
{
public:
  int dev {};
  size_t size {};
  size_t padded_size {};

  CUmemGenericAllocationHandle alloc_handle;

public:
  physical_memory () = delete;
  physical_memory (const physical_memory<data_type> &) = delete;
  physical_memory<data_type> operator= (const physical_memory<data_type> &) = delete;

  physical_memory (size_t size_arg, int dev_arg = 0)
    : dev (dev_arg)
    , size (size_arg * sizeof (data_type))
  {
    allocate ();
  }

  ~physical_memory ()
  {
    chck (cuMemRelease (alloc_handle));
  }

private:
  void allocate ()
  {
    const size_t granularity = get_granularity (dev);
    padded_size = round_up (size, granularity);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    chck (cuMemCreate(&alloc_handle, padded_size, &prop, 0));
  }
};

template <typename data_type>
class virtual_memory
{
public:
  size_t size {};
  size_t padded_size {};

  CUdeviceptr ptr;

public:
  virtual_memory (size_t size_arg, int dev = 0)
    : size (size_arg * sizeof (data_type))
  {
    allocate (dev);
  }

  ~virtual_memory ()
  {
    chck (cuMemAddressFree (ptr, padded_size));
  }

  data_type *get () { return (data_type*) ptr; }

private:
  void allocate (int dev)
  {
    const size_t granularity = get_granularity (dev);
    padded_size = round_up (size, granularity);
    chck (cuMemAddressReserve (&ptr, padded_size, 0, 0, 0));
  }
};

template <typename data_type>
class memory_mapper
{
  const virtual_memory<data_type> &virt;

public:
  memory_mapper (
    const virtual_memory<data_type> &virt_arg,
    const physical_memory<data_type> &phys_arg,
    const std::vector<int> &mapping_devices,
    unsigned int chunk)
    : virt (virt_arg)
  {
    const size_t size = phys_arg.padded_size;
    const size_t offset = size * chunk;
    chck (cuMemMap (virt.ptr + offset, size, 0, phys_arg.alloc_handle, 0));

    std::vector<CUmemAccessDesc> access_descriptors (mapping_devices.size ());

    for (unsigned int id = 0; id < mapping_devices.size (); id++)
      {
        access_descriptors[id].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_descriptors[id].location.id = mapping_devices[id];
        access_descriptors[id].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      }

    chck (cuMemSetAccess(virt.ptr + offset, size, access_descriptors.data (), access_descriptors.size ()));
  }

  ~memory_mapper ()
  {
    chck (cuMemUnmap (virt.ptr, virt.padded_size));
  }
};

template <typename data_type>
class memory
{
  physical_memory<data_type> phys;
  virtual_memory<data_type> virt;
  memory_mapper<data_type> mapper;

public:
  memory () = delete;

  memory (size_t n, int gpu, const std::vector<int> &mapping_devices)
    : phys (n, gpu)
    , virt (n, gpu)
    , mapper (virt, phys, mapping_devices, 0)
  { }

  CUdeviceptr &get_ptr () { return virt.ptr; }

  data_type *get () { return virt.get (); }
};

float basic_copy (size_t size)
{
  int *pointers[2];

  cudaSetDevice (0);
  cudaMalloc (&pointers[0], size);

  cudaSetDevice (1);
  cudaMalloc (&pointers[1], size);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);
  cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
  cudaEventRecord (end);
  cudaEventSynchronize (end);

  float elapsed;
  cudaEventElapsedTime (&elapsed, begin, end);
  elapsed /= 1000;

  cudaSetDevice (0);
  cudaFree (pointers[0]);

  cudaSetDevice (1);
  cudaFree (pointers[1]);

  cudaEventDestroy (end);
  cudaEventDestroy (begin);

  return elapsed;
}

float p2p_copy (size_t size)
{
  int *pointers[2];

  cudaSetDevice (0);
  cudaDeviceEnablePeerAccess (1, 0);
  cudaMalloc (&pointers[0], size);

  cudaSetDevice (1);
  cudaDeviceEnablePeerAccess (0, 0);
  cudaMalloc (&pointers[1], size);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);
  cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
  cudaEventRecord (end);
  cudaEventSynchronize (end);

  float elapsed;
  cudaEventElapsedTime (&elapsed, begin, end);
  elapsed /= 1000;

  cudaSetDevice (0);
  cudaFree (pointers[0]);

  cudaSetDevice (1);
  cudaFree (pointers[1]);

  cudaEventDestroy (end);
  cudaEventDestroy (begin);

  return elapsed;
}

__global__ void copy_kernel (const unsigned int n, const int * __restrict__ in, int * __restrict__ out)
{
  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n)
    out[i] = in[i];
}

float p2p_copy_kernel (size_t size)
{
  int *pointers[2];

  cudaSetDevice (0);
  cudaDeviceEnablePeerAccess (1, 0);
  cudaMalloc (&pointers[0], size);

  cudaSetDevice (1);
  cudaDeviceEnablePeerAccess (0, 0);
  cudaMalloc (&pointers[1], size);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);

  const unsigned int n = size / sizeof (int);
  const unsigned int block_size = 256;
  const unsigned int blocks_count = (n + block_size - 1) / block_size;

  copy_kernel<<<blocks_count, block_size>>>(n, pointers[0], pointers[1]);

  cudaEventRecord (end);
  cudaEventSynchronize (end);

  float elapsed;
  cudaEventElapsedTime (&elapsed, begin, end);
  elapsed /= 1000;

  cudaSetDevice (0);
  cudaFree (pointers[0]);

  cudaSetDevice (1);
  cudaFree (pointers[1]);

  cudaEventDestroy (end);
  cudaEventDestroy (begin);

  return elapsed;
}

float cumem_copy (size_t size)
{
  CUdevice device;
  CUcontext context;
  chck (cuDeviceGet (&device, 1));
  chck (cuCtxCreate (&context, 0, device));

  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  std::vector<int> devices (devices_count);
  std::iota (devices.begin (), devices.end (), 0);

  const size_t n = size / sizeof (int);

  memory<int> gpu_0_mem (n, 0, devices);
  memory<int> gpu_1_mem (n, 0, devices);

  chck (cuMemsetD32 (gpu_0_mem.get_ptr (), 0, n));
  chck (cuMemsetD32 (gpu_1_mem.get_ptr (), 0, n));

  CUevent begin, end;
  chck (cuEventCreate (&begin, 0));
  chck (cuEventCreate (&end, 0));

  cuEventRecord (begin, 0);
  chck (cuMemcpyDtoDAsync (gpu_0_mem.get_ptr (), gpu_1_mem.get_ptr (), size, 0));
  cuEventRecord (end, 0);
  cuEventSynchronize (end);

  float elapsed {};
  cuEventElapsedTime (&elapsed, begin, end);

  cuEventDestroy (end);
  cuEventDestroy (begin);
  cuCtxDestroy (context);

  return elapsed / 1000;
}

void print_bw (size_t size, float elapsed)
{
  const double size_gb = size / 1E9;
  std::cout << size << ", " << size_gb / elapsed << "\n";
}

int main ()
{
  cuInit (0);

  int attribute_val_0 {};
  int attribute_val_1 {};

  cuDeviceGetAttribute (&attribute_val_0, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, 0);
  cuDeviceGetAttribute (&attribute_val_1, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, 1);

  if (!attribute_val_0 || !attribute_val_1)
    std::cout << "One of the GPUs doesn't support virtual address management\n";

  for (size_t size: { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 524288, 2097152, 8388608, 33554432, 134217728 })
    {
      float elapsed {};
      const unsigned int max_iterations = 30;
      for (unsigned int _ = 0; _ < max_iterations; _++)
        elapsed += p2p_copy_kernel (size);
      print_bw (size, elapsed / max_iterations);
    }

  // print_bw (size, cumem_copy (size));
  // print_bw (size, p2p_copy (size));

  return 0;
}
