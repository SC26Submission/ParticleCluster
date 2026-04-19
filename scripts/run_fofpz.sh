#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

dataset=""
compressor=""
eps=""
qp=""
dimensions=("xx" "yy" "zz")
precision="f32"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d) dataset="$2"; shift 2 ;;
        -c) compressor="$2"; shift 2 ;;
        -eps) eps="$2"; shift 2 ;;
        -qp) qp="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$dataset" ]]; then
    echo "Error: -d is required"
    exit 1
fi
if [[ -z "$compressor" ]]; then
    echo "Error: -c is required"
    exit 1
fi
if [[ -z "$eps" && -z "$qp" ]]; then
    echo "Error: one of -eps and -qp is required"
    exit 1
fi

case "${dataset}" in
    "HACC_low")
        size=280953867
        step=1
        val_range=64
        ;;
    "EXAALT")
        size=2869440
        step=1
        val_range=51.8184
        ;;
    "FPM_hi")
        size=0
        step=60
        val_range=10
        ;;
    "FPM_mid")
        size=0
        step=121
        val_range=10
        ;;
    "FPM_low")
        size=0
        step=121
        val_range=10
        ;;
    *)
        echo "Error: Unknown dataset='$dataset'"
        echo "Valid datasets: HACC_low, EXAALT, FPM_hi, FPM_mid, FPM_low"
        exit 1
        ;;
esac

mkdir -p "$ROOT_DIR/results/${dataset}"

input_path="$ROOT_DIR/datasets/${dataset}/"
output_path="$ROOT_DIR/results/${dataset}/"
if [ -z "$qp" ]; then
    base_stat_file=$output_path"${compressor}.${eps}.txt"
    edit_stat_file=$output_path"${compressor}.${eps}.edit.txt"
    abs_err=$(awk "BEGIN {printf \"%.10g\", $eps * $val_range}")
else
    base_stat_file=$output_path"${compressor}.${qp}.txt"
    edit_stat_file=$output_path"${compressor}.${qp}.edit.txt"
fi

case "${compressor}" in
    "sz3")
        if [ "$step" -eq 1 ]; then
            for dim in "${dimensions[@]}"; do
                base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
                if [ ! -f "$base_decomp_file" ]; then
                    input_file=$input_path"${dim}.${precision}"
                    sz3 -${precision:0:1} -i $input_file -o $base_decomp_file -1 $size -M ABS $abs_err >> $base_stat_file 2>&1
                fi
            done
        else
            for i in $(seq 0 $((step-1))); do
                echo "===== Step ${i} =====" >> $base_stat_file
                for dim in "${dimensions[@]}"; do
                    base_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.out"
                    if [ ! -f "$base_decomp_file" ]; then
                        input_file=$input_path"${dim}.${i}.${precision}"
                        size=$(( $(stat -c%s "$input_file") / 4 ))
                        sz3 -${precision:0:1} -i $input_file -o $base_decomp_file -1 $size -M ABS $abs_err >> $base_stat_file 2>&1
                    fi
                done
            done
        fi
        ;;
    "cuszp")
        if [ "$step" -eq 1 ]; then
            for dim in "${dimensions[@]}"; do
                base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
                if [ ! -f "$base_decomp_file" ]; then
                    input_file=$input_path"${dim}.${precision}"
                    cuSZp -i $input_file -t $precision -m plain -d 1 -eb abs $abs_err -o $base_decomp_file >> $base_stat_file 2>&1
                fi
            done
        else
            for i in $(seq 0 $((step-1))); do
                echo "===== Step ${i} =====" >> $base_stat_file
                for dim in "${dimensions[@]}"; do
                    base_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.out"
                    if [ ! -f "$base_decomp_file" ]; then
                        input_file=$input_path"${dim}.${i}.${precision}"
                        cuSZp -i $input_file -t $precision -m plain -d 1 -eb abs $abs_err -o $base_decomp_file >> $base_stat_file 2>&1
                    fi
                done
            done
        fi
        ;;
    "zfp")
        if [ "$step" -eq 1 ]; then
            for dim in "${dimensions[@]}"; do
                base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
                if [ ! -f "$base_decomp_file" ]; then
                    input_file=$input_path"${dim}.${precision}"
                    temp_path=$output_path"${dim}.${compressor}.${eps}.tmp"
                    perf stat zfp -${precision:0:1} -i $input_file -z $temp_path -1 $size -a $abs_err >> $base_stat_file 2>&1
                    perf stat zfp -${precision:0:1} -z $temp_path -o $base_decomp_file -1 $size -a $abs_err >> $base_stat_file 2>&1
                    rm $temp_path
                fi
            done
        else
            for i in $(seq 0 $((step-1))); do
                echo "===== Step ${i} =====" >> $base_stat_file
                for dim in "${dimensions[@]}"; do
                    base_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.out"
                    temp_path=$output_path"${dim}.${i}.${compressor}.${eps}.tmp"
                    if [ ! -f "$base_decomp_file" ]; then
                        input_file=$input_path"${dim}.${i}.${precision}"
                        size=$(( $(stat -c%s "$input_file") / 4 ))
                        perf stat zfp -${precision:0:1} -i $input_file -z $temp_path -1 $size -a $abs_err >> $base_stat_file 2>&1
                        perf stat zfp -${precision:0:1} -z $temp_path -o $base_decomp_file -1 $size -a $abs_err >> $base_stat_file 2>&1
                        rm $temp_path
                    fi
                done
            done
        fi
        ;;
    "draco")
        if [ "$step" -eq 1 ]; then
            base_decomp_file=$output_path"xx.${compressor}.${qp}.out"
            temp_path=$output_path"${compressor}.${qp}.tmp"
            if [ ! -f "$base_decomp_file" ]; then
                input_file=$input_path"pts.ply"
                if [ ! -f "$input_file" ]; then
                    python3 analysis/bin2ply.py --d $dataset
                fi
                base_decomp_file=$output_path"pts.${compressor}.${qp}.ply"
                draco_encoder-1.5.7 -point_cloud -i $input_file -o $temp_path -qp $qp -cl 10 >> $base_stat_file 2>&1
                draco_decoder-1.5.7 -i $temp_path -o $base_decomp_file >> $base_stat_file 2>&1
                rm $temp_path
            fi
        else
            for i in $(seq 0 $((step-1))); do
                base_decomp_file=$output_path"xx.${i}.${compressor}.${qp}.out"
                temp_path=$output_path"${i}.${compressor}.${qp}.tmp"
                if [ ! -f "$base_decomp_file" ]; then
                    input_file=$input_path"pts.${i}.ply"
                    if [ ! -f "$input_file" ]; then
                        python3 analysis/bin2ply.py --d $dataset
                    fi
                    base_decomp_file=$output_path"pts.${i}.${compressor}.${qp}.ply"
                    echo "===== Step ${i} =====" >> $base_stat_file
                    draco_encoder-1.5.7 -point_cloud -i $input_file -o $temp_path -qp $qp -cl 10 >> $base_stat_file 2>&1
                    draco_decoder-1.5.7 -i $temp_path -o $base_decomp_file >> $base_stat_file 2>&1
                    rm $temp_path
                fi
            done
        fi
        ;;
    "lcp")
        if [ "$step" -eq 1 ]; then
            base_decomp_file=$output_path"xx.${compressor}.${eps}.out"
            if [ ! -f "$base_decomp_file" ]; then
                input_args="-i"
                for dim in "${dimensions[@]}"; do
                    input_file=$input_path"${dim}.${precision}"
                    input_args+=" $input_file"
                done
                temp_path=$output_path"${compressor}.${eps}.tmp"
                lcp $input_args -z $temp_path -osn -1 $size -eb $abs_err -a >> $base_stat_file 2>&1
                for dim in "${dimensions[@]}"; do
                    base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
                    mv "${dim:0:1}.lcp.out" $base_decomp_file
                done
                rm $temp_path
            fi
        elif [ "$size" -eq 0 ]; then
            temp_path=$output_path"${compressor}.${eps}.tmp"
            for i in $(seq 0 $((step-1))); do
                base_decomp_file=$output_path"xx.${i}.${compressor}.${eps}.out"
                if [ ! -f "$base_decomp_file" ]; then
                    input_args="-i"
                    for dim in "${dimensions[@]}"; do
                        input_file=$input_path"${dim}.${i}.${precision}"
                        input_args+=" $input_file"
                    done
                    size=$(( $(stat -c%s "$input_file") / 4 ))
                    echo "===== Step ${i} =====" >> $base_stat_file
                    lcp $input_args -z $temp_path -osn -1 $size -eb $abs_err -a >> $base_stat_file 2>&1
                    for dim in "${dimensions[@]}"; do
                        base_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.out"
                        mv "${dim:0:1}.lcp.out" $base_decomp_file
                    done
                    rm $temp_path
                fi
            done
        else
            base_decomp_file=$output_path"xx.${compressor}.${eps}.out"
            if [ ! -f "$base_decomp_file" ]; then
                input_args="-i"
                for dim in "${dimensions[@]}"; do
                    input_file=$input_path"${dim}.${precision}"
                    input_args+=" $input_file"
                done
                temp_path=$output_path"${compressor}.${eps}.tmp"
                lcp $input_args -z $temp_path -osn -2 $step $size -eb $abs_err -a >> $base_stat_file 2>&1
                for dim in "${dimensions[@]}"; do
                    base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
                    mv "${dim:0:1}.lcp.out" $base_decomp_file
                done
                rm $temp_path
            fi
        fi
        ;;
    "fofpz")
        if [ "$step" -eq 1 ]; then
            base_decomp_file=$output_path"xx.${compressor}.${eps}.out"
            if [ ! -f "$base_decomp_file" ]; then
                input_args="-i"
                for dim in "${dimensions[@]}"; do
                    input_file=$input_path"${dim}.${precision}"
                    input_args+=" $input_file"
                done
                temp_path=$output_path"${dim}.${compressor}.${eps}.tmp"
                fofpz $input_args -z $temp_path -o ${temp_path}. -N $size -M REL $eps -${precision:0:1} -c >> $base_stat_file 2>&1
                for dim in "${dimensions[@]}"; do
                    base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
                    mv "${temp_path}.${dim}.out" $base_decomp_file
                done
                fofpz $input_args -z $temp_path -o ${temp_path}. -N $size -M REL $eps -${precision:0:1} -lr 0.7 >> $edit_stat_file 2>&1
                for dim in "${dimensions[@]}"; do
                    edit_decomp_file=$output_path"${dim}.${compressor}.${eps}.edit"
                    mv "${temp_path}.${dim}.out" $edit_decomp_file
                done
                rm $temp_path
            fi
        else
            for i in $(seq 0 $((step-1))); do
                base_decomp_file=$output_path"xx.${i}.${compressor}.${eps}.out"
                if [ ! -f "$base_decomp_file" ]; then
                    input_args="-i"
                    for dim in "${dimensions[@]}"; do
                        input_file=$input_path"${dim}.${i}.${precision}"
                        input_args+=" $input_file"
                    done
                    size=$(( $(stat -c%s "$input_file") / 4 ))
                    temp_path=$output_path"${i}.${compressor}.${eps}.tmp"
                    echo "===== Step ${i} =====" >> $base_stat_file
                    fofpz $input_args -z $temp_path -o ${temp_path}. -N $size -M REL $eps -${precision:0:1} -c >> $base_stat_file 2>&1
                    for dim in "${dimensions[@]}"; do
                        base_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.out"
                        mv "${temp_path}.${dim}.out" $base_decomp_file
                    done
                    echo "===== Step ${i} =====" >> $edit_stat_file
                    fofpz $input_args -z $temp_path -o ${temp_path}. -N $size -M REL $eps -${precision:0:1} -lr 0.7 >> $edit_stat_file 2>&1
                    for dim in "${dimensions[@]}"; do
                        edit_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.edit"
                        mv "${temp_path}.${dim}.out" $edit_decomp_file
                    done
                    rm $temp_path
                fi
            done
        fi
        ;;
    *)
        echo "Error: Unknown compressor='$compressor'"
        echo "Valid compressors: sz3, zfp, cuszp, draco, lcp, fofpz"
        exit 1
        ;;
esac

if [ $compressor != "fofpz" ]; then
    if [ "$step" -eq 1 ]; then
        input_args="-i"
        base_args="-e"
        for dim in "${dimensions[@]}"; do
            input_file=$input_path"${dim}.${precision}"
            input_args+=" $input_file"
            if [ -z "$qp" ]; then
                base_decomp_file=$output_path"${dim}.${compressor}.${eps}.out"
            else
                base_decomp_file=$output_path"${dim}.${compressor}.${qp}.out"
            fi
            base_args+=" $base_decomp_file"
        done
        size=$(( $(stat -c%s "$input_file") / 4 ))
        temp_path=$output_path"${dim}.${compressor}.${eps}.tmp"
        fofpz $input_args $base_args -z $temp_path -o ${temp_path}. -N $size -M REL $eps -${precision:0:1} -lr 0.7 >> $edit_stat_file 2>&1
        for dim in "${dimensions[@]}"; do
            if [ -z "$qp" ]; then
                edit_decomp_file=$output_path"${dim}.${compressor}.${eps}.edit"
            else
                edit_decomp_file=$output_path"${dim}.${compressor}.${qp}.edit"
            fi
            mv "${temp_path}.${dim}.out" $edit_decomp_file
        done
        rm $temp_path
    else
        for i in $(seq 0 $((step-1))); do
            if [ -z "$qp" ]; then
                edit_check_file=$output_path"xx.${i}.${compressor}.${eps}.edit"
            else
                edit_check_file=$output_path"xx.${i}.${compressor}.${qp}.edit"
            fi
            if [ ! -f "$edit_check_file" ]; then
                input_args="-i"
                base_args="-e"
                for dim in "${dimensions[@]}"; do
                    input_file=$input_path"${dim}.${i}.${precision}"
                    input_args+=" $input_file"
                    if [ -z "$qp" ]; then
                        base_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.out"
                    else
                        base_decomp_file=$output_path"${dim}.${i}.${compressor}.${qp}.out"
                    fi
                    base_args+=" $base_decomp_file"
                done
                size=$(( $(stat -c%s "$input_file") / 4 ))
                temp_path=$output_path"${dim}.${i}.${compressor}.tmp"
                echo "===== Step ${i} =====" >> $edit_stat_file
                fofpz $input_args $base_args -z $temp_path -o ${temp_path}. -N $size -M REL $eps -${precision:0:1} -lr 0.7 >> $edit_stat_file 2>&1
                for dim in "${dimensions[@]}"; do
                    if [ -z "$qp" ]; then
                        edit_decomp_file=$output_path"${dim}.${i}.${compressor}.${eps}.edit"
                    else
                        edit_decomp_file=$output_path"${dim}.${i}.${compressor}.${qp}.edit"
                    fi
                    mv "${temp_path}.${dim}.out" $edit_decomp_file
                done
                rm $temp_path
            fi
        done
    fi
fi