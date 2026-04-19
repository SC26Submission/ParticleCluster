import numpy as np

if __name__ == "__main__":
    # valid_steps = [100, 102, 107, 113, 121, 127, 134, 141, 144, 148, 151, 159, 167, 180, 184, 189, 198, 213, 219, 224, 230, 235, 241, 247, 253]
    valid_steps = [100, 102, 107, 113, 121, 127, 134, 141, 144, 148, 159, 167, 180, 184, 189, 198, 213, 219, 224, 230, 235, 241, 247, 253]
    R = 64
    for step in valid_steps:
        fpath = "/pscratch/sd/c/crren/datasets/HACC_" + str(step) + "/m000.full.mpicosmo." + str(step) + "#"
        for i in range(R):
            xx = np.fromfile(fpath + str(i) + '-0.dat', np.float32)
            yy = np.fromfile(fpath + str(i) + '-1.dat', np.float32)
            zz = np.fromfile(fpath + str(i) + '-2.dat', np.float32)
            mask = (xx == 0) & (yy == 0) & (zz == 0)
            xx = xx[~mask]
            yy = yy[~mask]
            zz = zz[~mask]
            xx.tofile(fpath + str(i) + '-0.dat')
            yy.tofile(fpath + str(i) + '-1.dat')
            zz.tofile(fpath + str(i) + '-2.dat')
        print(step)
