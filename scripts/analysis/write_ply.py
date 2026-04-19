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


if __name__ == "__main__":
    #  eval_steps = [121, 127, 134, 141, 144, 148, 151, 159, 167, 180, 184, 189, 198, 213, 219, 224]
    eval_steps = [134, 141, 144, 148, 151, 159, 167, 180, 184, 189, 198, 213, 219, 224]
    R = 64
    for step in eval_steps:
        fpath = "/pscratch/sd/c/crren/datasets/HACC_" + str(step) + "/"
        fprefix = fpath + "m000.full.mpicosmo." + str(step) + "#"
        for i in range(R):
            xx = np.fromfile(fprefix + str(i) + '-0.dat', np.float32)
            n = xx.shape[0]
            pts = np.zeros((n, 3), dtype=np.float32)
            pts[:, 0] = xx
            pts[:, 1] = np.fromfile(fprefix + str(i) + '-1.dat', np.float32)
            pts[:, 2] = np.fromfile(fprefix + str(i) + '-2.dat', np.float32)
            plyWriteParticles(fpath + str(i) + "-pts.ply", pts)
            print("step:", step, "; rank:", i)
