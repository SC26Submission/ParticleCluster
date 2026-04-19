#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --nodes=64
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:03:00
#SBATCH -A <YOUR_NERSC_ACCOUNT>
#SBATCH -q regular
#SBATCH -o output.%j
#SBATCH -e error.%j

# OPTIONAL: Uncomment and fill in your email to get notifications
# #SBATCH --mail-type=begin,end,fail
# #SBATCH --mail-user=<YOUR_EMAIL@DOMAIN.COM>

module reset
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80

export PATH="$HOME/particle/GPU/build:$PATH"
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_RX_MATCH_MODE=software

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

compressor="sz3"
eps="1e-6"
num_ranks=256

DATA_DIR=/pscratch/sd/c/crren/datasets/HACC_split151_256
RESULT_DIR=/pscratch/sd/c/crren/results/HACC_split151_256
result_dir=/pscratch/sd/c/crren/results/HACC_split151_256/gpu_mpi_profile

out_dir="${result_dir}/output"
mkdir -p "$result_dir" "$out_dir"

input_args="-i $DATA_DIR/xx.%r.f32 $DATA_DIR/yy.%r.f32 $DATA_DIR/zz.%r.f32"
base_args="-e $RESULT_DIR/xx.%r.${compressor}.${eps}.f32 $RESULT_DIR/yy.%r.${compressor}.${eps}.f32 $RESULT_DIR/zz.%r.${compressor}.${eps}.f32"

fofpz_args="-f $input_args $base_args -O $out_dir -M REL $eps -iter 100 -lr 0.7"

prefix="${result_dir}/${compressor}.${eps}.r${num_ranks}"

# ── 1. Baseline timing: [Timer] from all ranks ──────────────────────────────
srun --export=ALL --label \
	fofpz_mpi $fofpz_args \
	>"${prefix}.log" 2>"${prefix}.err"

# ── 2. nsys: multi-rank GPU timeline with MPI tracing ───────────────────────
# Correct pattern at NERSC: "srun nsys profile ..." — nsys wraps each rank's
# process directly. %q{SLURM_PROCID} expands per-rank so each rank writes its
# own .nsys-rep file: prefix_0.nsys-rep, prefix_1.nsys-rep, ...
dcgmi profile --pause

srun --export=ALL \
	nsys profile \
	--mpi-impl=mpich \
	--trace=cuda,mpi,nvtx \
	--force-overwrite=true \
	-o "${prefix}_%q{SLURM_PROCID}.nsys" \
	-- fofpz_mpi $fofpz_args \
	>"${prefix}.nsys.log" 2>&1

# Export kernel timeline and MPI summary from rank 0
nsys stats "${prefix}_0.nsys.nsys-rep" \
	--report cuda_gpu_trace --format csv --force-export=true \
	>"${prefix}.nsys.r0.cuda.csv" 2>&1
nsys stats "${prefix}_0.nsys.nsys-rep" \
	--report mpi_event_summary --format csv --force-export=true \
	>"${prefix}.nsys.r0.mpi.csv" 2>&1

# ── 3. ncu: per-kernel roofline metrics on rank 0 only ──────────────────────
# Write wrapper to SLURM_SUBMIT_DIR (shared Lustre), NOT /tmp (node-local).
# NCU_OUTPUT is passed via environment so the wrapper can use the correct path.
WRAPPER="${SCRIPT_DIR}/ncu_rank0_wrapper_${SLURM_JOB_ID}.sh"
cat >"$WRAPPER" <<WRAPPER_EOF
#!/bin/bash
if [ "\$SLURM_PROCID" -eq 0 ]; then
  exec ncu -f -o "\${NCU_OUTPUT}" \
    --metrics gpu__time_duration.sum,dram__bytes.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
    fofpz_mpi "\$@"
else
  exec fofpz_mpi "\$@"
fi
WRAPPER_EOF
chmod +x "$WRAPPER"

NCU_OUTPUT="${prefix}.ncu" \
	srun --export=ALL \
	"$WRAPPER" $fofpz_args \
	>"${prefix}.ncu.log" 2>&1

ncu -i "${prefix}.ncu.ncu-rep" --csv >"${prefix}.ncu.csv" 2>&1

rm -f "$WRAPPER"
dcgmi profile --resume
