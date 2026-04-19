"""
This file converts all binary files into .ply files as inputs of Google Draco.
"""

import numpy as np
import argparse
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent


def plyWriteParticles(fpath: str, pts: np.array, scalar_fields: dict = {}):
    points = vtk.vtkPoints()
    n_pts, dim = pts.shape
    if dim == 2:
        for pt in pts:
            points.InsertNextPoint(pt[0], pt[1], 0)
    else:
        for pt in pts:
            points.InsertNextPoint(pt[0], pt[1], pt[2])

    cells = vtk.vtkCellArray()
    cell = vtk.vtkVertex()
    for i in range(n_pts):
        cell.GetPointIds().SetId(0, i)
        cells.InsertNextCell(cell)

    data_save = vtk.vtkPolyData()
    data_save.SetPoints(points)
    data_save.SetPolys(cells)

    pd = data_save.GetPointData()
    for i, (k, v) in enumerate(scalar_fields.items()):
        vtk_array = numpy_to_vtk(v)
        vtk_array.SetName(k)
        if i == 0:
            pd.SetScalars(vtk_array)
        else:
            pd.AddArray(vtk_array)

    writer = vtk.vtkPLYWriter()
    writer.SetInputData(data_save)
    writer.SetFileName(fpath)
    writer.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    args = parser.parse_args()
    dataset = args.d
    fpath = ROOT_DIR / "datasets" / dataset

    num_bin_files = len(list(fpath.glob("xx*.f32")))

    fpath = str(fpath) + "/"

    if num_bin_files == 1:
        size = os.path.getsize(fpath + "xx.f32") // 4
        pts = np.zeros((size, 3), dtype=np.float32)
        pts[:, 0] = np.fromfile(fpath + "xx.f32")
        pts[:, 1] = np.fromfile(fpath + "yy.f32")
        pts[:, 2] = np.fromfile(fpath + "zz.f32")
        plyWriteParticles(fpath + "pts.ply")
    else:
        for i in range(num_bin_files):
            size = os.path.getsize(fpath + "xx." + str(i) + ".f32") // 4
            pts = np.zeros((size, 3), dtype=np.float32)
            pts[:, 0].fromfile(fpath + "xx." + str(i) + ".f32")
            pts[:, 1].fromfile(fpath + "yy." + str(i) + ".f32")
            pts[:, 2].fromfile(fpath + "zz." + str(i) + ".f32")
            plyWriteParticles(fpath + "pts." + str(i) + ".ply")
