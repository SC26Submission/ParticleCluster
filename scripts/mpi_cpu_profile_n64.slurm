#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=64
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH -A <YOUR_NERSC_ACCOUNT>
#SBATCH -q regular
#SBATCH -o output.%j
#SBATCH -e error.%j

# # OPTIONAL: Uncomment and fill in your email to get notifications
# #SBATCH --mail-type=begin,end,fail
# #SBATCH --mail-user=<YOUR_EMAIL@DOMAIN.COM>

module reset
module load PrgEnv-gnu

export PATH="$HOME/particle/CPU/build:$PATH"
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_RX_MATCH_MODE=software
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

compressor="sz3"
eps="1e-6"
num_ranks=256

DATA_DIR=/pscratch/sd/c/crren/datasets/HACC_split151_256
RESULT_DIR=/pscratch/sd/c/crren/results/HACC_split151_256
result_dir=/pscratch/sd/c/crren/results/HACC_split151_256/cpu_mpi_profile2

mkdir -p "$result_dir"

input_args="-i $DATA_DIR/xx.%r.f32 $DATA_DIR/yy.%r.f32 $DATA_DIR/zz.%r.f32"
base_args="-e $RESULT_DIR/xx.%r.${compressor}.${eps}.f32 $RESULT_DIR/yy.%r.${compressor}.${eps}.f32 $RESULT_DIR/zz.%r.${compressor}.${eps}.f32"

fofpz_args="-f $input_args $base_args -O $result_dir -M REL $eps -iter 100 -lr 0.7"

prefix="${result_dir}/${compressor}.${eps}.r${num_ranks}"

# ── 1. Baseline timing: [Timer] from all ranks ──────────────────────────────
# --label prefixes each stdout line with "rank: " for per-rank timing and
# load imbalance analysis.
srun --export=ALL --label \
	fofpz_cpu_mpi $fofpz_args \
	>"${prefix}.log" 2>"${prefix}.err"

# ── 2. perf stat on rank 0 only ─────────────────────────────────────────────
# Write wrapper to SLURM_SUBMIT_DIR (shared Lustre), NOT /tmp (node-local).
# All compute nodes can read from the submit directory.
WRAPPER="${SCRIPT_DIR}/perf_stat_wrapper_${SLURM_JOB_ID}.sh"
cat >"$WRAPPER" <<WRAPPER_EOF
#!/bin/bash
if [ "\$SLURM_PROCID" -eq 0 ]; then
  exec perf stat -e cycles:u,instructions:u,fp_ret_sse_avx_ops.all:u,cache-misses:u,cache-references:u \
    -- fofpz_cpu_mpi "\$@"
else
  exec fofpz_cpu_mpi "\$@"
fi
WRAPPER_EOF
chmod +x "$WRAPPER"

srun --export=ALL \
	"$WRAPPER" $fofpz_args \
	>"${prefix}.perf.log" 2>"${prefix}.perf.stat.txt"

rm -f "$WRAPPER"

# ── 3. perf record on rank 0 only (cycles, FP ops, cache misses) ────────────
for evt in cycles fp_ret_sse_avx_ops.all cache-misses; do
	WRAPPER="${SCRIPT_DIR}/perf_record_wrapper_${SLURM_JOB_ID}_${evt}.sh"
	cat >"$WRAPPER" <<WRAPPER_EOF
#!/bin/bash
if [ "\$SLURM_PROCID" -eq 0 ]; then
  exec perf record -g -e "${evt}:u" -o "${prefix}.${evt}.data" -- fofpz_cpu_mpi "\$@"
else
  exec fofpz_cpu_mpi "\$@"
fi
WRAPPER_EOF
	chmod +x "$WRAPPER"

	srun --export=ALL \
		"$WRAPPER" $fofpz_args \
		>"${prefix}.${evt}.log" 2>&1

	perf report --stdio -i "${prefix}.${evt}.data" >"${prefix}.${evt}.txt"
	rm -f "$WRAPPER"
done
