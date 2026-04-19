#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH -A <YOUR_NERSC_ACCOUNT>       # FIXME: REPLACE WITH YOUR NERSC ALLOCATION
#SBATCH -q regular                    # Use 'debug' for faster queue times if testing
#SBATCH -o output.%j                  # standard output (stdout)
#SBATCH -e error.%j                   # standard error (stderr)

# OPTIONAL: Uncomment and fill in your email to get notifications
# #SBATCH --mail-type=begin,end,fail
# #SBATCH --mail-user=<YOUR_EMAIL@DOMAIN.COM>

export PATH="$HOME/SZ3/build/tools/sz3:$PATH"
export PATH="$HOME/zfp/build/bin:$PATH"
export PATH="$HOME/draco/build:$PATH"
export PATH="$HOME/LCP/build/bin:$PATH"

set -euo pipefail
ulimit -s unlimited
export OMP_NUM_THREADS=1

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

datasets=("FPM_mid")
# datasets=("HACC_mid" "EXAALT_MD" "EXAALT_Copper" "EXAALT_Helium" "FPM_high" "FPM_mid" "FPM_low")
error_bounds=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 1e-6 1e-7 1e-8 1e-9)
# quant_lengths=(2 4 6 8 10 12 14 16 18 20 22 24 26 28)
quant_lengths=(8 26)

while getopts ":p" opt; do
  case $opt in
    p)
      plot=true
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

for i in "${!datasets[@]}"; do
    # for epsilon in "${error_bounds[@]}"; do
    #     echo "base: SZ3, data: ${datasets[$i]}, error bound: $epsilon"
    #     bash "$SCRIPT_DIR/run_fofpz.sh" -d ${datasets[$i]} -c sz3 -eps $epsilon
    #     echo "base: ZFP, data: ${datasets[$i]}, error bound: $epsilon"
    #     bash "$SCRIPT_DIR/run_fofpz.sh" -d ${datasets[$i]} -c zfp -eps $epsilon
    #     echo "base: LCP, data: ${datasets[$i]}, error bound: $epsilon"
    #     bash "$SCRIPT_DIR/run_fofpz.sh" -d ${datasets[$i]} -c lcp -eps $epsilon
    # done
    for quant_len in "${quant_lengths[@]}"; do
        echo "base: Draco, data: ${datasets[$i]}, quant length: $quant_len"
        bash "$SCRIPT_DIR/run_fofpz.sh" -d ${datasets[$i]} -c draco -qp $quant_len
    done
done
