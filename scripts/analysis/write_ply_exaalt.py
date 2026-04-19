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


def separateMultipleSteps(fpath_x, fpath_y, fpath_z, step, size, out_path):
    pts = np.zeros((step, size, 3), dtype=np.float32)
    pts[:,:, 0] = np.fromfile(fpath_x, np.float32).reshape(step, size)
    pts[:,:, 1] = np.fromfile(fpath_y, np.float32).reshape(step, size)
    pts[:,:, 2] = np.fromfile(fpath_z, np.float32).reshape(step, size)
    for i in range(step):
        plyWriteParticles(out_path + "pts." + str(i) + ".ply", pts[i])
        pts[i,:, 0].tofile(out_path + "xx." + str(i) + ".f32")
        pts[i,:, 1].tofile(out_path + "yy." + str(i) + ".f32")
        pts[i,:, 2].tofile(out_path + "zz." + str(i) + ".f32")
        print(i, step)


if __name__ == "__main__":
    fpath = "/pscratch/sd/c/crren/datasets/EXAALT_Copper/"
    separateMultipleSteps(fpath + "xx.f32", fpath + "yy.f32", fpath + "zz.f32", 83, 1077290, fpath)
    fpath = "/pscratch/sd/c/crren/datasets/EXAALT_Helium/"
    separateMultipleSteps(fpath + "xx.f32", fpath + "yy.f32", fpath + "zz.f32", 2338, 106711, fpath)
