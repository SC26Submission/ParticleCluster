import numpy as np

if __name__ == "__main__":
    split = np.arange(9, dtype=int) * 32
    
    xx = np.fromfile("../datasets/HACC_mid/xx.f32", np.float32)
    yy = np.fromfile("../datasets/HACC_mid/yy.f32", np.float32)
    zz = np.fromfile("../datasets/HACC_mid/zz.f32", np.float32)
    base_xx = np.fromfile("../results/HACC_mid/xx.lcp.1e-7.out", np.float32)
    base_yy = np.fromfile("../results/HACC_mid/yy.lcp.1e-7.out", np.float32)
    base_zz = np.fromfile("../results/HACC_mid/zz.lcp.1e-7.out", np.float32)
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            pt_idx = np.where((yy >= split[i]) & (yy < split[i + 1]) & (zz >= split[j]) & (zz < split[j + 1]))[0]
            xx[pt_idx].tofile("../datasets/HACC_mid_split/xx." + str(idx) + ".f32")
            yy[pt_idx].tofile("../datasets/HACC_mid_split/yy." + str(idx) + ".f32")
            zz[pt_idx].tofile("../datasets/HACC_mid_split/zz." + str(idx) + ".f32")
            base_xx[pt_idx].tofile("../results/HACC_mid_split/xx." + str(idx) + ".lcp.1e-7.out")
            base_yy[pt_idx].tofile("../results/HACC_mid_split/yy." + str(idx) + ".lcp.1e-7.out")
            base_zz[pt_idx].tofile("../results/HACC_mid_split/zz." + str(idx) + ".lcp.1e-7.out")
