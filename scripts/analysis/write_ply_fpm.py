import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def plyWriteParticles(fpath: str, pts: np.array, scalar_fields: dict={}):
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


def separateMultipleSteps(fpath, step, out_path):
    for i in range(step):
        xx = np.fromfile(fpath + "xx." + str(i) + ".f32", np.float32)
        n = xx.shape[0]
        pts = np.zeros((n, 3), dtype=np.float32)
        pts[:, 0] = xx
        pts[:, 1] = np.fromfile(fpath + "yy." + str(i) + ".f32", np.float32)
        pts[:, 2] = np.fromfile(fpath + "zz." + str(i) + ".f32", np.float32)
        plyWriteParticles(out_path + "pts." + str(i) + ".ply", pts)
        print(i, step)


if __name__ == "__main__":
    fpath = "/pscratch/sd/c/crren/datasets/FPM_high/"
    separateMultipleSteps(fpath, 60, fpath)
    fpath = "/pscratch/sd/c/crren/datasets/FPM_mid/"
    separateMultipleSteps(fpath, 121, fpath)
    fpath = "/pscratch/sd/c/crren/datasets/FPM_low/"
    separateMultipleSteps(fpath, 121, fpath)
