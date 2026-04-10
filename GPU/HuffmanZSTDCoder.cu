#include "HuffmanZSTDCoder.cuh"

__device__ void atomicOrByte(uint8_t *address, uint8_t val) {
  unsigned int *base_addr = (unsigned int *)((size_t)address & ~3);
  unsigned int byte_offset = (size_t)address & 3;
  unsigned int shift = byte_offset * 8;
  unsigned int mask = (unsigned int)val << shift;
  atomicOr(base_addr, mask);
}

__global__ void packBytesAligned(const uint32_t *__restrict__ codes,
                                 const uint8_t *__restrict__ lengths,
                                 const size_t *__restrict__ bit_positions,
                                 uint8_t *__restrict__ output, size_t N) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  uint32_t code = codes[idx];
  uint8_t len = lengths[idx];
  size_t bit_start = bit_positions[idx];

  for (int i = 0; i < len; ++i) {
    size_t bit_pos = bit_start + i;
    size_t byte_idx = bit_pos >> 3;
    int bit_idx = 7 - (bit_pos & 7);

    if ((code >> (len - 1 - i)) & 1) {
      atomicOrByte(&output[byte_idx], (uint8_t)(1 << bit_idx));
    }
  }
}
