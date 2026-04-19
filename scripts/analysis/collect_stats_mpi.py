import numpy as np
import pickle
from pathlib import Path
import os
import re

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

R = 64
val_range = 256
metrics_type = [("cr", float), ("bpp", float), ("comp_time", float), ("decomp_time", float), ("comp_throughput", float), ("max_ae", float), ("mse", float), ("rmse", float), ("nrmse", float), ("psnr", float)]
edit_metrics_type = [("vulnerable", int), ("editable", int), ("confusion_l_before", (int, 4)), ("mcc_before", float), ("confusion_h_before", (int, 4)), ("ari_before", float), ("iter", int), ("loss", float), ("confusion_l_after", (int, 4)), ("mcc_after", float), ("confusion_h_after", (int, 4)), ("ari_after", float), ("cr", float), ("bpp", float), ("comp_time", (float, R)), ("comp_throughput", (float, R)), ("max_ae", float), ("mse", float), ("rmse", float), ("nrmse", float), ("psnr", float), ("local", (int, R)), ("ghost", (int, R))]
ours_metrics_type = [("cr", float), ("bpp", float), ("comp_time", (float, R)), ("decomp_time", float), ("comp_throughput", (float, R)), ("max_ae", float), ("mse", float), ("rmse", float), ("nrmse", float), ("psnr", float)]

# error_bounds = ["1e-2", "5e-3", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"]
# quant_lengths = ["2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28"]
error_bounds = ["1e-3", "5e-4", "1e-4", "5e-5", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"]
quant_lengths = ["8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28"]
num_err_bounds = len(error_bounds)
num_quant_lens = len(quant_lengths)
start_keys = [ "compression time = ", "compression ratio = ", "decompression time = ", "ratio=", "", "", "compression ratio = ", "compression time = ", "decompression time = ", " (", "Encoded size = ", " (", "PCompression ratio: ", "Algorithm total execution time (ms): ", "Decompression_CUDA execution time (ms): ", "cuSZp compression   end-to-end speed: ", "cuSZp decompression end-to-end speed: ", "cuSZp compression ratio: "]
end_keys = ["\n", "\n", " seconds", " ", " seconds user", " seconds user", " \n", "\n", "\n", " ms", " bytes", " ms", "\n", "\n", "\n", " GB/s", " GB/s", "\n"]
edit_start_keys = ["Number of vulnerable pairs: ", "Number of editable particles: ", "Before editing: TP_l = ", "TN_l = ", "FP_l = ", "FN_l = ", "MCC = ", "TP_h = ", "TN_h = ", "FP_h = ", "FN_h = ", "ARI = ", "Number of iterations: ", "PGD final loss: ", "After editing: TP_l = ", "Edit time: ", "Compression time: ", "Additional storage: ", "Compression ratio: ", "BPP: ", "MAE: ", "MSE: ", "RMSE: ", "NRMSE: ", "PSNR: "]
edit_end_keys = ["\n", "\n", ",", ",", ",", ",", "\n", ",", ",", ",", ",", "\n", "\n", "\n", ",", " seconds", " seconds", " bytes", "\n", "\n", "\n", "\n", "\n", "\n", "dB"]


def extract_between(text, start_key, end_key, start_id=0):
    if start_key:
        start_idx = text.find(start_key, start_id)
        if start_idx == -1:
            return None, None
        start_idx += len(start_key)
        end_idx = text.find(end_key, start_idx)
        return text[start_idx:end_idx], end_idx
    else:
        end_idx = text.find(end_key, start_id)
        if end_idx == -1:
            return None, None
        segment = text[:end_idx].rstrip()
        return segment.split()[-1], end_idx


def extract_edit(compressor, result_fpath):
    with open(f"{result_fpath}{compressor}_metrics.pickle", "rb") as f:
        metrics = pickle.load(f)
    if compressor in ["draco", "cuzfp"]:
        params = quant_lengths
    else:
        params = error_bounds
    edit_metrics = np.zeros(len(params), dtype=edit_metrics_type)
    for i, p in enumerate(params):
        if not os.path.exists(f"{result_fpath}{compressor}.{p}.edit.txt"):
            print("edit, compressor:", compressor, "error / quant:", p)
            break
        with open(f"{result_fpath}{compressor}.{p}.edit.txt", "r") as f:
            content = f.read()
        
        pt_counts = re.findall(r"Rank (\d+):\s+(\d+)\s+local,\s+(\d+)\s+ghost", content)
        for rank_id, local, ghost in pt_counts:
            rank_id = int(rank_id)
            edit_metrics["local"][i][rank_id] = int(local)
            edit_metrics["ghost"][i][rank_id] = int(ghost)
            
        block_pattern = r"Grid coarsened:.*?Rank (\d+): stage10_after_edit"
        blocks = re.finditer(block_pattern, content, re.DOTALL)
        additional_storage = 0
        org_storage = 0
        for match in blocks:
            rank_id = int(match.group(1))
            stat = match.group(0)
            
            local_storage = os.path.getsize(result_fpath + str(rank_id) + "-0." + compressor + "." + p + ".out")
            org_storage += local_storage
            
            tmp, idx = extract_between(stat, edit_start_keys[0], edit_end_keys[0], 0)
            num_vulnerable = int(tmp)
            edit_metrics["vulnerable"][i] += num_vulnerable
            tmp, idx = extract_between(stat, edit_start_keys[1], edit_end_keys[1], idx)
            edit_metrics["editable"][i] += int(tmp)
            for j in range(4):
                tmp, idx = extract_between(stat, edit_start_keys[j + 2], edit_end_keys[j + 2], idx)
                edit_metrics["confusion_l_before"][i][j] += int(tmp)
            tmp, idx = extract_between(stat, edit_start_keys[12], edit_end_keys[12], idx)
            edit_metrics["iter"][i] = int(tmp)
            if num_vulnerable > 0:
                tmp, idx = extract_between(stat, edit_start_keys[14], edit_end_keys[14], idx)
                edit_metrics["confusion_l_after"][i][0] += int(tmp)
                for j in range(1, 4):
                    tmp, idx = extract_between(stat, edit_start_keys[j + 2], edit_end_keys[j + 2], idx)
                    edit_metrics["confusion_l_after"][i][j] += int(tmp)
            tmp, idx = extract_between(stat, edit_start_keys[15], edit_end_keys[15], idx)
            edit_metrics["comp_time"][i][rank_id] = float(tmp)
            edit_metrics["comp_throughput"][i][rank_id] = local_storage / 2 ** 30 / edit_metrics["comp_time"][i][rank_id]
            tmp, idx = extract_between(stat, edit_start_keys[17], edit_end_keys[17], idx)
            additional_storage += float(tmp)
            tmp, idx = extract_between(stat, edit_start_keys[20], edit_end_keys[20], idx)
            edit_metrics["max_ae"][i] = max(float(tmp), edit_metrics["max_ae"][i])
            tmp, idx = extract_between(stat, edit_start_keys[21], edit_end_keys[21], idx)
            edit_metrics["mse"][i] += local_storage * 3 / 4 * float(tmp)
            
        org_storage_gb = org_storage / 2 ** 30
        edit_metrics["cr"][i] = org_storage / (org_storage / metrics["cr"][i] + additional_storage)
        edit_metrics["bpp"][i] = 3 * 32 / edit_metrics["cr"][i]
        edit_metrics["mse"][i] /= (org_storage * 3 / 4)
        edit_metrics["rmse"][i] = np.sqrt(edit_metrics["mse"][i])
        edit_metrics["nrmse"][i] = edit_metrics["rmse"][i] / val_range
        edit_metrics["psnr"][i] = -20 * np.log10(edit_metrics["nrmse"][i])
            
    with open(f"{result_fpath}{compressor}_edit_metrics.pickle", "wb") as f:
        pickle.dump(edit_metrics, f)


if __name__ == "__main__":
    # Collect base statistics
    orig_fpath = str(ROOT_DIR / "datasets/HACC_151/m000.full.mpicosmo.151#") 
    result_fpath = str(ROOT_DIR / "results/HACC_151") + "/"
    orig_storage = 0
    total_size = 0
    # sz3_metrics = np.zeros(num_err_bounds, dtype=metrics_type)
    # zfp_metrics = np.zeros(num_err_bounds, dtype=metrics_type)
    # lcp_metrics = np.zeros(num_err_bounds, dtype=metrics_type)
    # cuszp_metrics = np.zeros(num_err_bounds, dtype=metrics_type)
    # cuzfp_metrics = np.zeros(num_quant_lens, dtype=metrics_type)
    # draco_metrics = np.zeros(num_quant_lens, dtype=metrics_type)
    # ours_metrics = np.zeros(num_err_bounds, dtype=ours_metrics_type)
    # for r in range(R):
    #     orig_file = orig_fpath + str(r) + "-0.dat" 
    #     local_storage = os.path.getsize(orig_file)
    #     orig_storage += local_storage * 3
    #     size = int(local_storage / 4)
    #     total_size += size
    #     orig_data = np.zeros((size, 3), dtype=np.float32)
    #     for d in range(3):
    #         orig_file = orig_fpath + str(r) + "-" + str(d) + ".dat" 
    #         orig_data[:, d] = np.fromfile(orig_file, np.float32)
    #         for i, err in enumerate(error_bounds):
    #             ############### SZ3 ###############
    #             decomp_file = result_fpath + str(r) + "-" + str(d) + ".sz3." + err + ".out"
    #             diff = np.fromfile(decomp_file, np.float32)
    #             diff -= orig_data[:, d]
    #             max_ae = np.abs(diff).max()
    #             if max_ae > sz3_metrics["max_ae"][i]:
    #                 sz3_metrics["max_ae"][i] = max_ae
    #                 sz3_metrics["mse"][i] += np.sum(diff ** 2)
    #             ############### ZFP ###############
    #             decomp_file = result_fpath + str(r) + "-" + str(d) + ".zfp." + err + ".out"
    #             diff = np.fromfile(decomp_file, np.float32)
    #             diff -= orig_data[:, d]
    #             max_ae = np.abs(diff).max()
    #             if max_ae > zfp_metrics["max_ae"][i]:
    #                 zfp_metrics["max_ae"][i] = max_ae
    #                 zfp_metrics["mse"][i] += np.sum(diff ** 2)
    #             ############### LCP ###############
    #             decomp_file = result_fpath + str(r) + "-" + str(d) + ".lcp." + err + ".out"
    #             diff = np.fromfile(decomp_file, np.float32)
    #             diff -= orig_data[:, d]
    #             max_ae = np.abs(diff).max()
    #             if max_ae > lcp_metrics["max_ae"][i]:
    #                 lcp_metrics["max_ae"][i] = max_ae
    #                 lcp_metrics["mse"][i] += np.sum(diff ** 2)
    #             ############### cuSZp2 ###############
    #             decomp_file = result_fpath + str(r) + "-" + str(d) + ".cuszp." + err + ".out"
    #             diff = np.fromfile(decomp_file, np.float32)
    #             diff -= orig_data[:, d]
    #             max_ae = np.abs(diff).max()
    #             if max_ae > cuszp_metrics["max_ae"][i]:
    #                 cuszp_metrics["max_ae"][i] = max_ae
    #                 cuszp_metrics["mse"][i] += np.sum(diff ** 2)
    #         for i, qp in enumerate(quant_lengths):
    #             ############### Draco ###############
    #             decomp_file = result_fpath + str(r) + "-" + str(d) + ".draco." + qp + ".out"
    #             diff = np.fromfile(decomp_file, np.float32)
    #             diff -= orig_data[:, d]
    #             max_ae = np.abs(diff).max()
    #             if max_ae > draco_metrics["max_ae"][i]:
    #                 draco_metrics["max_ae"][i] = max_ae
    #                 draco_metrics["mse"][i] += np.sum(diff ** 2)
    #     print("Rank", r, "MSE calculated!")
    
    # orig_storage_gb = orig_storage / 2 ** 30
    # for i, err in enumerate(error_bounds):
    #     ############### SZ3 ###############
    #     cr = 0
    #     idx = 0
    #     with open(f"{result_fpath}sz3.{err}.txt", "r") as f:
    #         stat = f.read()
    #     for r in range(R):
    #         orig_file = orig_fpath + str(r) + "-0.dat" 
    #         local_storage = os.path.getsize(orig_file)
    #         for _ in range(3):
    #             tmp, idx = extract_between(stat, start_keys[0], end_keys[0], idx)
    #             if tmp is None:
    #                 print("base, compressor: sz3 error:", err, "step:", r)
    #                 break
    #             sz3_metrics["comp_time"][i] += float(tmp)
    #             tmp, idx = extract_between(stat, start_keys[1], end_keys[1], idx)
    #             if tmp is None:
    #                 print("base, compressor: sz3 error:", err, "step:", r)
    #                 break
    #             cr += local_storage / float(tmp)
    #             tmp, idx = extract_between(stat, start_keys[2], end_keys[2], idx)
    #             if tmp is None:
    #                 print("base, compressor: sz3 error:", err, "step:", r)
    #                 break
    #             sz3_metrics["decomp_time"][i] += float(tmp)
    #     sz3_metrics["cr"][i] = orig_storage / cr
    #     sz3_metrics["bpp"][i] = 32 * 3 / sz3_metrics["cr"][i]
    #     sz3_metrics["comp_throughput"][i] = orig_storage_gb / sz3_metrics["comp_time"][i]
    #     sz3_metrics["mse"][i] /= (orig_storage / 4)
    #     sz3_metrics["rmse"][i] = np.sqrt(sz3_metrics["mse"][i])
    #     sz3_metrics["nrmse"][i] = sz3_metrics["rmse"][i] / val_range
    #     sz3_metrics["psnr"][i] = -20 * np.log10(sz3_metrics["nrmse"][i])
    #     ############### ZFP ###############
    #     cr = 0
    #     idx = 0
    #     with open(f"{result_fpath}zfp.{err}.txt", "r") as f:
    #         stat = f.read()
    #     for r in range(R):
    #         orig_file = orig_fpath + str(r) + "-0.dat" 
    #         local_storage = os.path.getsize(orig_file)
    #         for _ in range(3):
    #             tmp, idx = extract_between(stat, start_keys[3], end_keys[3], idx)
    #             if tmp is None:
    #                 print("base, compressor: zfp error:", err, "step:", r)
    #                 break
    #             cr += local_storage / float(tmp)
    #             tmp, idx = extract_between(stat, start_keys[4], end_keys[4], idx)
    #             if tmp is None:
    #                 print("base, compressor: zfp error:", err, "step:", r)
    #                 break
    #             zfp_metrics["comp_time"][i] += float(tmp)
    #             tmp, idx = extract_between(stat, start_keys[5], end_keys[5], idx)
    #             if tmp is None:
    #                 print("base, compressor: zfp error:", err, "step:", r)
    #                 break
    #             zfp_metrics["decomp_time"][i] += float(tmp)
    #     zfp_metrics["cr"][i] = orig_storage / cr
    #     zfp_metrics["bpp"][i] = 32 * 3 / zfp_metrics["cr"][i]
    #     zfp_metrics["comp_throughput"][i] = orig_storage_gb / zfp_metrics["comp_time"][i]
    #     zfp_metrics["mse"][i] /= (orig_storage / 4)
    #     zfp_metrics["rmse"][i] = np.sqrt(zfp_metrics["mse"][i])
    #     zfp_metrics["nrmse"][i] = zfp_metrics["rmse"][i] / val_range
    #     zfp_metrics["psnr"][i] = -20 * np.log10(zfp_metrics["nrmse"][i])
    #     ############### LCP ###############
    #     cr = 0
    #     idx = 0
    #     with open(f"{result_fpath}lcp.{err}.txt", "r") as f:
    #         stat = f.read()
    #     for r in range(R):
    #         orig_file = orig_fpath + str(r) + "-0.dat" 
    #         local_storage = os.path.getsize(orig_file)
    #         tmp, idx = extract_between(stat, start_keys[6], end_keys[6], idx)
    #         if tmp is None:
    #             print("base, compressor: lcp error:", err, "step:", r)
    #             break
    #         cr += 3 * local_storage / float(tmp)
    #         tmp, idx = extract_between(stat, start_keys[7], end_keys[7], idx)
    #         if tmp is None:
    #             print("base, compressor: lcp error:", err, "step:", r)
    #             break
    #         lcp_metrics["comp_time"][i] += float(tmp)
    #         tmp, idx = extract_between(stat, start_keys[8], end_keys[8], idx)
    #         if tmp is None:
    #             print("base, compressor: lcp error:", err, "step:", r)
    #             break
    #         lcp_metrics["decomp_time"][i] += float(tmp)
    #     lcp_metrics["cr"][i] = orig_storage / cr
    #     lcp_metrics["bpp"][i] = 3 * 32 / lcp_metrics["cr"][i]
    #     lcp_metrics["comp_throughput"][i] = orig_storage_gb / lcp_metrics["comp_time"][i]
    #     lcp_metrics["mse"][i] /= (orig_storage / 4)
    #     lcp_metrics["rmse"][i] = np.sqrt(lcp_metrics["mse"][i])
    #     lcp_metrics["nrmse"][i] = lcp_metrics["rmse"][i] / val_range
    #     lcp_metrics["psnr"][i] = -20 * np.log10(lcp_metrics["nrmse"][i])
    #     ############### cuSZp2 ###############
    #     cr = 0
    #     idx = 0
    #     with open(f"{result_fpath}cuszp.{err}.txt", "r") as f:
    #         stat = f.read()
    #     for r in range(R):
    #         orig_file = orig_fpath + str(r) + "-0.dat" 
    #         local_storage = os.path.getsize(orig_file)
    #         for _ in range(3):
    #             tmp, idx = extract_between(stat, start_keys[15], end_keys[15], idx)
    #             if tmp is None:
    #                 print("base, compressor: cuszp error:", err, "step:", r)
    #                 idx = len(stat)
    #                 break
    #             cuszp_metrics["comp_time"][i] += orig_storage_gb / float(tmp)
    #             tmp, idx = extract_between(stat, start_keys[16], end_keys[16], idx)
    #             if tmp is None:
    #                 print("base, compressor: cuszp error:", err, "step:", r)
    #                 idx = len(stat)
    #                 break
    #             cuszp_metrics["decomp_time"][i] += orig_storage_gb / float(tmp)
    #             tmp, idx = extract_between(stat, start_keys[17], end_keys[17], idx)
    #             if tmp is None:
    #                 print("base, compressor: cuszp error:", err, "step:", r)
    #                 idx = len(stat)
    #                 break
    #             cr += local_storage / float(tmp)
    #     cuszp_metrics["cr"][i] = orig_storage / cr
    #     cuszp_metrics["bpp"][i] = 32 * 3 / cuszp_metrics["cr"][i]
    #     cuszp_metrics["comp_throughput"][i] = orig_storage_gb / cuszp_metrics["comp_time"][i]
    #     cuszp_metrics["mse"][i] /= (orig_storage / 4)
    #     cuszp_metrics["rmse"][i] = np.sqrt(cuszp_metrics["mse"][i])
    #     cuszp_metrics["nrmse"][i] = cuszp_metrics["rmse"][i] / val_range
    #     cuszp_metrics["psnr"][i] = -20 * np.log10(cuszp_metrics["nrmse"][i])
        
    #     ############### ours ###############
    #     if not os.path.exists(f"{result_fpath}fofpz.{err}.txt"):
    #         print("base, txt not exist, compressor: fofpz err:", err)
    #         break
    #     with open(f"{result_fpath}fofpz.{err}.txt", "r") as f:
    #         content = f.read()
        
    #     block_pattern = r"Grid coarsened:.*?Rank (\d+):"
    #     blocks = re.finditer(block_pattern, content, re.DOTALL)
    #     cr = 0
    #     for match in blocks:
    #         rank_id = int(match.group(1))
    #         stat = match.group(0)
            
    #         orig_file = orig_fpath + str(rank_id) + "-0.dat" 
    #         local_storage = os.path.getsize(orig_file) * 3
            
    #         tmp, idx = extract_between(stat, edit_start_keys[16], edit_end_keys[16], 0)
    #         ours_metrics["comp_time"][i][rank_id] = float(tmp)
    #         ours_metrics["comp_throughput"][i][rank_id] = local_storage / 2 ** 30 / ours_metrics["comp_time"][i][rank_id]
    #         tmp, idx = extract_between(stat, edit_start_keys[18], edit_end_keys[18], idx)
    #         cr += local_storage / float(tmp)
    #         tmp, idx = extract_between(stat, edit_start_keys[20], edit_end_keys[20], idx)
    #         ours_metrics["max_ae"][i] = max(float(tmp), ours_metrics["max_ae"][i])
    #         tmp, idx = extract_between(stat, edit_start_keys[21], edit_end_keys[21], idx)
    #         ours_metrics["mse"][i] += local_storage * 3 / 4 * float(tmp)
            
    #     ours_metrics["cr"][i] = orig_storage / cr
    #     ours_metrics["bpp"][i] = 3 * 32 / ours_metrics["cr"][i]
    #     ours_metrics["mse"][i] /= (orig_storage / 4)
    #     ours_metrics["rmse"][i] = np.sqrt(ours_metrics["mse"][i])
    #     ours_metrics["nrmse"][i] = ours_metrics["rmse"][i] / val_range
    #     ours_metrics["psnr"][i] = -20 * np.log10(ours_metrics["nrmse"][i])
        
    #     print(i, ": error", err, "base calculated!")
        
    # for i, qp in enumerate(quant_lengths):
    #     compressed_size = 0
    #     idx = 0
    #     ############### Draco ###############
    #     with open(f"{result_fpath}draco.{qp}.txt", "r") as f:
    #         stat = f.read()
    #     for s in range(R):
    #         tmp, idx = extract_between(stat, start_keys[9], end_keys[9], 0)
    #         if tmp is None:
    #             print("base, compressor: draco quant:", qp, "step:", s)
    #             break
    #         draco_metrics["comp_time"][i] += float(tmp) * 1e-3
    #         tmp, idx = extract_between(stat, start_keys[10], end_keys[10], idx)
    #         if tmp is None:
    #             print("base, compressor: draco quant:", qp, "step:", s)
    #             break
    #         compressed_size += float(tmp)
    #         tmp, idx = extract_between(stat, start_keys[11], end_keys[11], idx)
    #         if tmp is None:
    #             print("base, compressor: draco quant:", qp, "step:", s)
    #             break
    #         draco_metrics["decomp_time"][i] += float(tmp) * 1e-3
    #     draco_metrics["cr"][i] = orig_storage / compressed_size
    #     draco_metrics["bpp"][i] = 3 * 32 / draco_metrics["cr"][i]
    #     draco_metrics["comp_throughput"][i] = orig_storage_gb / draco_metrics["comp_time"][i]
    #     draco_metrics["rmse"][i] = np.sqrt(draco_metrics["mse"][i])
    #     draco_metrics["nrmse"][i] = draco_metrics["rmse"][i] / val_range
    #     draco_metrics["psnr"][i] = -20 * np.log10(draco_metrics["nrmse"][i])
    #     ############### cuZFP ###############
    #     idx = 0
    #     with open(f"{result_fpath}cuzfp.{qp}.txt", "r") as f:
    #         stat = f.read()
    #     for _ in range(3):
    #         tmp, idx = extract_between(stat, start_keys[4], end_keys[4], idx)
    #         cuzfp_metrics["comp_time"][i] += float(tmp)
    #         tmp, idx = extract_between(stat, start_keys[5], end_keys[5], idx)
    #         cuzfp_metrics["decomp_time"][i] += float(tmp)
    #     cuzfp_metrics["comp_throughput"][i] = orig_storage_gb / cuzfp_metrics["comp_time"][i]
    
    #     print(i, ": quant", qp, "base calculated!")
    
    # with open(f"{result_fpath}sz3_metrics.pickle", "wb") as f:
    #     pickle.dump(sz3_metrics, f)
    # with open(f"{result_fpath}zfp_metrics.pickle", "wb") as f:
    #     pickle.dump(zfp_metrics, f)
    # with open(f"{result_fpath}lcp_metrics.pickle", "wb") as f:
    #     pickle.dump(lcp_metrics, f)
    # with open(f"{result_fpath}cuszp_metrics.pickle", "wb") as f:
    #     pickle.dump(cuszp_metrics, f)
    # with open(f"{result_fpath}draco_metrics.pickle", "wb") as f:
    #     pickle.dump(draco_metrics, f)
    # with open(f"{result_fpath}cuzfp_metrics.pickle", "wb") as f:
    #     pickle.dump(cuzfp_metrics, f)
    # with open(f"{result_fpath}ours_metrics.pickle", "wb") as f:
    #     pickle.dump(ours_metrics, f)

    # Collect edit statistics
    ############### SZ3 ###############
    extract_edit("sz3", result_fpath)
    print("SZ3 edit done!")
    ############### ZFP ###############
    extract_edit("zfp", result_fpath)
    print("ZFP edit done!")
    ############### cuZSp2 ###############
    extract_edit("cuszp", result_fpath)
    print("cuSZp2 edit done!")
    ############### Draco ###############
    extract_edit("draco", result_fpath)
    print("Draco edit done!")
    ############### LCP ###############
    extract_edit("lcp", result_fpath)
    print("LCP edit done!")
    ############### ours ###############
    with open(f"{result_fpath}ours_metrics.pickle", "rb") as f:
        metrics = pickle.load(f)
    edit_metrics = np.zeros(len(error_bounds), dtype=edit_metrics_type)
    for i, err in enumerate(error_bounds):
        if not os.path.exists(f"{result_fpath}fofpz.{err}.edit.txt"):
            print("edit, txt not exist, compressor: fofpz err:", err)
            break
        with open(f"{result_fpath}fofpz.{err}.edit.txt", "r") as f:
            content = f.read()
        
        block_pattern = r"Grid coarsened:.*?Rank (\d+): stage10_after_edit"
        blocks = re.finditer(block_pattern, content, re.DOTALL)
        cr = 0
        for match in blocks:
            rank_id = int(match.group(1))
            stat = match.group(0)
            
            local_storage = os.path.getsize(orig_fpath + str(rank_id) + "-0.dat") * 3
            
            tmp, idx = extract_between(stat, edit_start_keys[0], edit_end_keys[0], 0)
            num_vulnerable = int(tmp)
            edit_metrics["vulnerable"][i] += num_vulnerable
            tmp, idx = extract_between(stat, edit_start_keys[1], edit_end_keys[1], idx)
            edit_metrics["editable"][i] += int(tmp)
            for j in range(4):
                tmp, idx = extract_between(stat, edit_start_keys[j + 2], edit_end_keys[j + 2], idx)
                edit_metrics["confusion_l_before"][i][j] += int(tmp)
            tmp, idx = extract_between(stat, edit_start_keys[12], edit_end_keys[12], idx)
            edit_metrics["iter"][i] = int(tmp)
            if num_vulnerable > 0:
                tmp, idx = extract_between(stat, edit_start_keys[14], edit_end_keys[14], idx)
                edit_metrics["confusion_l_after"][i][0] += int(tmp)
                for j in range(1, 4):
                    tmp, idx = extract_between(stat, edit_start_keys[j + 2], edit_end_keys[j + 2], idx)
                    edit_metrics["confusion_l_after"][i][j] += int(tmp)
            tmp, idx = extract_between(stat, edit_start_keys[16], edit_end_keys[16], idx)
            edit_metrics["comp_time"][i][rank_id] = float(tmp)
            edit_metrics["comp_throughput"][i][rank_id] = local_storage / 2 ** 30 / edit_metrics["comp_time"][i][rank_id]
            tmp, idx = extract_between(stat, edit_start_keys[18], edit_end_keys[18], idx)
            cr += local_storage / float(tmp)
            tmp, idx = extract_between(stat, edit_start_keys[20], edit_end_keys[20], idx)
            edit_metrics["max_ae"][i] = max(float(tmp), edit_metrics["max_ae"][i])
            tmp, idx = extract_between(stat, edit_start_keys[21], edit_end_keys[21], idx)
            edit_metrics["mse"][i] += local_storage * 3 / 4 * float(tmp)
        
        if cr == 0:
            continue
        edit_metrics["cr"][i] = orig_storage / cr
        edit_metrics["bpp"][i] = 3 * 32 / edit_metrics["cr"][i]
        edit_metrics["mse"][i] /= (orig_storage / 4)
        edit_metrics["rmse"][i] = np.sqrt(edit_metrics["mse"][i])
        edit_metrics["nrmse"][i] = edit_metrics["rmse"][i] / val_range
        edit_metrics["psnr"][i] = -20 * np.log10(edit_metrics["nrmse"][i])
            
    with open(f"{result_fpath}ours_edit_metrics.pickle", "wb") as f:
        pickle.dump(edit_metrics, f)

    print("ours edit done!")
