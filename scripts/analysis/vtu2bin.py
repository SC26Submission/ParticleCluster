"""
This file converts all .vtu files in a folder to binary files.
"""

import numpy as np
import argparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent


def vtuReadParticles(fpath: str):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fpath)
    reader.Update()

    output = reader.GetOutput()
    vtk_pts = output.GetPoints().GetData()
    return vtk_to_numpy(vtk_pts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    args = parser.parse_args()
    dataset = args.d
    fpath = ROOT_DIR / "datasets" / dataset

    num_vtu_files = len(list(fpath.glob("*.vtu")))

    fpath = str(fpath) + "/"

    for i in range(num_vtu_files):
        pts = vtuReadParticles(fpath + str(i).zfill(3) + ".vtu")
        pts[:, 0].tofile(fpath + "xx." + str(i) + ".f32")
        pts[:, 1].tofile(fpath + "yy." + str(i) + ".f32")
        pts[:, 2].tofile(fpath + "zz." + str(i) + ".f32")
