import numpy as np
import shutil

if __name__ == "__main__":
    R = 64
    size = np.array([18073456, 16887737, 17800320, 15184406, 14422449, 16331594, 16196456, 14681238, 16810803, 16306342, 17280234, 15411249, 17144319, 16161624, 18249220, 15889506, 16497215, 16033017, 16735652, 17751584, 15134254, 15422593, 17516787, 16732204, 17625621, 15676701, 19087251, 15394359, 16536920, 18877598, 18122965, 15455233, 19054576, 16190177, 17580420, 15842792, 16979708, 16621171, 17516761, 15512431, 15914509, 17794307, 16839532, 17365584, 16027615, 17357963, 17330453, 17533140, 16042595, 14421978, 16295425, 17483113, 17118231, 15217466, 18465161, 15829716, 17884153, 16837463, 17738505, 15469517, 17991983, 17262235, 19404501, 17379927])
    compressor = "sz3"
    err = "1e-6"
    
    # reorganize data for weak scaling
    indices = [121, 127, 134, 141, 144, 148, 151, 159, 167, 180, 184, 189, 198, 213, 219, 224]

    for i, idx in enumerate(indices):
        for r in range(R):
            ii = i * R + r
            # xx = np.fromfile("../datasets/HACC_" + str(idx) + "/m000.full.mpicosmo." + str(idx) + "#" + str(r) + "-0.dat", np.float32)
            # xx += 256 * i
            # xx.tofile("../datasets/HACC_16steps/xx." + str(ii) + ".f32")
            # shutil.copy("../datasets/HACC_" + str(idx) + "/m000.full.mpicosmo." + str(idx) + "#" + str(r) + "-1.dat", "../datasets/HACC_16steps/yy." + str(ii) + ".f32")
            # shutil.copy("../datasets/HACC_" + str(idx) + "/m000.full.mpicosmo." + str(idx) + "#" + str(r) + "-2.dat", "../datasets/HACC_16steps/zz." + str(ii) + ".f32")
            
            base_xx = np.fromfile("../results/HACC_" + str(idx) + "/" + str(r) + "-0." + compressor + "." + err + ".out", np.float32)
            base_xx += 256 * i
            base_xx.tofile("../results/HACC_16steps/xx." + str(ii) + "." + compressor + "." + err + ".f32")
            shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-1." + compressor + "." + err + ".out", "../results/HACC_16steps/yy." + str(ii) + "." + compressor + "." + err + ".f32")
            shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-2." + compressor + "." + err + ".out", "../results/HACC_16steps/zz." + str(ii) + "." + compressor + "." + err + ".f32")
            if ii < 128:
                base_xx.tofile("../results/HACC_2steps/xx." + str(ii) + "." + compressor + "." + err + ".f32")
                shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-1." + compressor + "." + err + ".out", "../results/HACC_2steps/yy." + str(ii) + "." + compressor + "." + err + ".f32")
                shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-2." + compressor + "." + err + ".out", "../results/HACC_2steps/zz." + str(ii) + "." + compressor + "." + err + ".f32")
            elif ii < 256:
                base_xx.tofile("../results/HACC_4steps/xx." + str(ii) + "." + compressor + "." + err + ".f32")
                shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-1." + compressor + "." + err + ".out", "../results/HACC_4steps/yy." + str(ii) + "." + compressor + "." + err + ".f32")
                shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-2." + compressor + "." + err + ".out", "../results/HACC_4steps/zz." + str(ii) + "." + compressor + "." + err + ".f32")
            elif ii < 512:
                base_xx.tofile("../results/HACC_8steps/xx." + str(ii) + "." + compressor + "." + err + ".f32")
                shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-1." + compressor + "." + err + ".out", "../results/HACC_8steps/yy." + str(ii) + "." + compressor + "." + err + ".f32")
                shutil.copy("../results/HACC_" + str(idx) + "/" + str(r) + "-2." + compressor + "." + err + ".out", "../results/HACC_8steps/zz." + str(ii) + "." + compressor + "." + err + ".f32")
        print(i)
            
    # # reorganize data for strong scaling
    # idx = 151
    # for r in range(R):
    #     xx = np.fromfile("../datasets/HACC_" + str(idx) + "/m000.full.mpicosmo." + str(idx) + "#" + str(r) + "-0.dat", np.float32)
    #     sorted_idx = np.argsort(xx)
    #     diff = xx[sorted_idx[1:]] - xx[sorted_idx[:-1]]
    #     block_indices = np.zeros(5, dtype=int)
    #     block_indices[1:-1] = np.where(diff > 30)[0] + 1
    #     block_indices[-1] = size[r]
    #     yy = np.fromfile("../datasets/HACC_" + str(idx) + "/m000.full.mpicosmo." + str(idx) + "#" + str(r) + "-1.dat", np.float32)
    #     zz = np.fromfile("../datasets/HACC_" + str(idx) + "/m000.full.mpicosmo." + str(idx) + "#" + str(r) + "-2.dat", np.float32)
    #     base_xx = np.fromfile("../results/HACC_" + str(idx) + "/" + str(r) + "-0." + compressor + "." + err + ".out", np.float32)
    #     base_yy = np.fromfile("../results/HACC_" + str(idx) + "/" + str(r) + "-1." + compressor + "." + err + ".out", np.float32)
    #     base_zz = np.fromfile("../results/HACC_" + str(idx) + "/" + str(r) + "-2." + compressor + "." + err + ".out", np.float32)
    #     for i in range(4):
    #         ii = r * 4 + i
    #         xx[sorted_idx[block_indices[i]:block_indices[i + 1]]].tofile("../datasets/HACC_split151_256/xx." + str(ii) + ".f32")
    #         yy[sorted_idx[block_indices[i]:block_indices[i + 1]]].tofile("../datasets/HACC_split151_256/yy." + str(ii) + ".f32")
    #         zz[sorted_idx[block_indices[i]:block_indices[i + 1]]].tofile("../datasets/HACC_split151_256/zz." + str(ii) + ".f32")
    #         base_xx[sorted_idx[block_indices[i]:block_indices[i + 1]]].tofile("../results/HACC_split151_256/xx." + str(ii) + "." + compressor + "." + err + ".f32")
    #         base_yy[sorted_idx[block_indices[i]:block_indices[i + 1]]].tofile("../results/HACC_split151_256/yy." + str(ii) + "." + compressor + "." + err + ".f32")
    #         base_zz[sorted_idx[block_indices[i]:block_indices[i + 1]]].tofile("../results/HACC_split151_256/zz." + str(ii) + "." + compressor + "." + err + ".f32")
    #     for i in range(2):
    #         ii = r * 2 + i
    #         xx[sorted_idx[block_indices[2 * i]:block_indices[2 * (i + 1)]]].tofile("../datasets/HACC_split151_128/xx." + str(ii) + ".f32")
    #         yy[sorted_idx[block_indices[2 * i]:block_indices[2 * (i + 1)]]].tofile("../datasets/HACC_split151_128/yy." + str(ii) + ".f32")
    #         zz[sorted_idx[block_indices[2 * i]:block_indices[2 * (i + 1)]]].tofile("../datasets/HACC_split151_128/zz." + str(ii) + ".f32")
    #         base_xx[sorted_idx[block_indices[2 * i]:block_indices[2 * (i + 1)]]].tofile("../results/HACC_split151_128/xx." + str(ii) + "." + compressor + "." + err + ".f32")
    #         base_yy[sorted_idx[block_indices[2 * i]:block_indices[2 * (i + 1)]]].tofile("../results/HACC_split151_128/yy." + str(ii) + "." + compressor + "." + err + ".f32")
    #         base_zz[sorted_idx[block_indices[2 * i]:block_indices[2 * (i + 1)]]].tofile("../results/HACC_split151_128/zz." + str(ii) + "." + compressor + "." + err + ".f32")
    #     xx[sorted_idx].tofile("../datasets/HACC_split151_64/xx." + str(r) + ".f32")
    #     yy[sorted_idx].tofile("../datasets/HACC_split151_64/yy." + str(r) + ".f32")
    #     zz[sorted_idx].tofile("../datasets/HACC_split151_64/zz." + str(r) + ".f32")
    #     base_xx[sorted_idx].tofile("../results/HACC_split151_64/xx." + str(r) + "." + compressor + "." + err + ".f32")
    #     base_yy[sorted_idx].tofile("../results/HACC_split151_64/yy." + str(r) + "." + compressor + "." + err + ".f32")
    #     base_zz[sorted_idx].tofile("../results/HACC_split151_64/zz." + str(r) + "." + compressor + "." + err + ".f32")
    #     r_ii = r % 2
    #     r_jj = r // 2
    #     if r_ii == 0:
    #         xx_merge_2 = np.zeros(size[r:r + 2].sum(), dtype=np.float32)
    #         yy_merge_2 = np.zeros(size[r:r + 2].sum(), dtype=np.float32)
    #         zz_merge_2 = np.zeros(size[r:r + 2].sum(), dtype=np.float32)
    #         xx_merge_2[:size[r]] = xx[sorted_idx]
    #         yy_merge_2[:size[r]] = yy[sorted_idx]
    #         zz_merge_2[:size[r]] = zz[sorted_idx]
    #         base_xx_merge_2 = np.zeros(size[r:r + 2].sum(), dtype=np.float32)
    #         base_yy_merge_2 = np.zeros(size[r:r + 2].sum(), dtype=np.float32)
    #         base_zz_merge_2 = np.zeros(size[r:r + 2].sum(), dtype=np.float32)
    #         base_xx_merge_2[:size[r]] = base_xx[sorted_idx]
    #         base_yy_merge_2[:size[r]] = base_yy[sorted_idx]
    #         base_zz_merge_2[:size[r]] = base_zz[sorted_idx]
    #     else:
    #         xx_merge_2[size[r - 1]:] = xx[sorted_idx]
    #         yy_merge_2[size[r - 1]:] = yy[sorted_idx]
    #         zz_merge_2[size[r - 1]:] = zz[sorted_idx]
    #         xx_merge_2.tofile("../datasets/HACC_split151_32/xx." + str(r_jj) + ".f32")
    #         yy_merge_2.tofile("../datasets/HACC_split151_32/yy." + str(r_jj) + ".f32")
    #         zz_merge_2.tofile("../datasets/HACC_split151_32/zz." + str(r_jj) + ".f32")
    #         base_xx_merge_2[size[r - 1]:] = base_xx[sorted_idx]
    #         base_yy_merge_2[size[r - 1]:] = base_yy[sorted_idx]
    #         base_zz_merge_2[size[r - 1]:] = base_zz[sorted_idx]
    #         base_xx_merge_2.tofile("../results/HACC_split151_32/xx." + str(r_jj) + "." + compressor + "." + err + ".f32")
    #         base_yy_merge_2.tofile("../results/HACC_split151_32/yy." + str(r_jj) + "." + compressor + "." + err + ".f32")
    #         base_zz_merge_2.tofile("../results/HACC_split151_32/zz." + str(r_jj) + "." + compressor + "." + err + ".f32")
    #     r_ii = r % 4
    #     r_jj = r // 4
    #     if r_ii == 0:
    #         xx_merge_4 = np.zeros(size[r:r + 4].sum(), dtype=np.float32)
    #         yy_merge_4 = np.zeros(size[r:r + 4].sum(), dtype=np.float32)
    #         zz_merge_4 = np.zeros(size[r:r + 4].sum(), dtype=np.float32)
    #         xx_merge_4[:size[r]] = xx[sorted_idx]
    #         yy_merge_4[:size[r]] = yy[sorted_idx]
    #         zz_merge_4[:size[r]] = zz[sorted_idx]
    #         base_xx_merge_4 = np.zeros(size[r:r + 4].sum(), dtype=np.float32)
    #         base_yy_merge_4 = np.zeros(size[r:r + 4].sum(), dtype=np.float32)
    #         base_zz_merge_4 = np.zeros(size[r:r + 4].sum(), dtype=np.float32)
    #         base_xx_merge_4[:size[r]] = base_xx[sorted_idx]
    #         base_yy_merge_4[:size[r]] = base_yy[sorted_idx]
    #         base_zz_merge_4[:size[r]] = base_zz[sorted_idx]
    #     elif r_ii == 3:
    #         xx_merge_4[-size[r]:] = xx[sorted_idx]
    #         yy_merge_4[-size[r]:] = yy[sorted_idx]
    #         zz_merge_4[-size[r]:] = zz[sorted_idx]
    #         xx_merge_4.tofile("../datasets/HACC_split151_16/xx." + str(r_jj) + ".f32")
    #         yy_merge_4.tofile("../datasets/HACC_split151_16/yy." + str(r_jj) + ".f32")
    #         zz_merge_4.tofile("../datasets/HACC_split151_16/zz." + str(r_jj) + ".f32")
    #         base_xx_merge_4[-size[r]:] = base_xx[sorted_idx]
    #         base_yy_merge_4[-size[r]:] = base_yy[sorted_idx]
    #         base_zz_merge_4[-size[r]:] = base_zz[sorted_idx]
    #         base_xx_merge_4.tofile("../results/HACC_split151_16/xx." + str(r_jj) + "." + compressor + "." + err + ".f32")
    #         base_yy_merge_4.tofile("../results/HACC_split151_16/yy." + str(r_jj) + "." + compressor + "." + err + ".f32")
    #         base_zz_merge_4.tofile("../results/HACC_split151_16/zz." + str(r_jj) + "." + compressor + "." + err + ".f32")
    #     else:
    #         xx_merge_4[size[r - r_ii:r].sum():size[r - r_ii:r + 1].sum()] = xx[sorted_idx]
    #         yy_merge_4[size[r - r_ii:r].sum():size[r - r_ii:r + 1].sum()] = yy[sorted_idx]
    #         zz_merge_4[size[r - r_ii:r].sum():size[r - r_ii:r + 1].sum()] = zz[sorted_idx]
    #         base_xx_merge_4[size[r - r_ii:r].sum():size[r - r_ii:r + 1].sum()] = base_xx[sorted_idx]
    #         base_yy_merge_4[size[r - r_ii:r].sum():size[r - r_ii:r + 1].sum()] = base_yy[sorted_idx]
    #         base_zz_merge_4[size[r - r_ii:r].sum():size[r - r_ii:r + 1].sum()] = base_zz[sorted_idx]
            
    #     print(r)
            
