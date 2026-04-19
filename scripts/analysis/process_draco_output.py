import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os

# datasets = ["HACC_mid", "EXAALT_MD", "EXAALT_Copper", "EXAALT_Helium", "HACC_151", "FPM_high", "FPM_mid", "FPM_low"]
# steps = [1, 1, 83, 2338, 64, 60, 121, 121]
datasets = ["FPM_mid"]
steps = [121]
# qps = np.arange(14, dtype=int) * 2 + 2
qps = [8, 26]


def plyReadParticles(fpath:str):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(fpath)
    reader.Update()
    
    output = reader.GetOutput()
    vtk_pts = output.GetPoints().GetData()
    return vtk_to_numpy(vtk_pts)


if __name__ == "__main__": 
    fpath = "/pscratch/sd/c/crren/results/"
    for i, dataset in enumerate(datasets):
        step = steps[i]
        for qp in qps:
            if step == 1:
                fname = fpath + dataset + "/pts.draco." + str(qp) + ".ply"
                if os.path.exists(fname):
                    pts = plyReadParticles(fname)
                    order = np.fromfile(fname + ".order", np.uint32)
                    order = np.argsort(order)
                    pts[order, 0].tofile(fpath + dataset + "/xx.draco." + str(qp) + ".out")
                    pts[order, 1].tofile(fpath + dataset + "/yy.draco." + str(qp) + ".out")
                    pts[order, 2].tofile(fpath + dataset + "/zz.draco." + str(qp) + ".out")
                    os.remove(fname)
                    os.remove(fname + ".order")
            else:
                for s in range(step):
                    fname = fpath + dataset + "/pts." + str(s) + ".draco." + str(qp) + ".ply"
                    if os.path.exists(fname):
                        pts = plyReadParticles(fname)
                        order = np.fromfile(fname + ".order", np.uint32)
                        order = np.argsort(order)
                        pts[order, 0].tofile(fpath + dataset + "/xx." + str(s) + ".draco." + str(qp) + ".out")
                        pts[order, 1].tofile(fpath + dataset + "/yy." + str(s) + ".draco." + str(qp) + ".out")
                        pts[order, 2].tofile(fpath + dataset + "/zz." + str(s) + ".draco." + str(qp) + ".out")
                        os.remove(fname)
                        os.remove(fname + ".order")
                    fname = fpath + dataset + "/" + str(s) + "-pts.draco." + str(qp) + ".ply"
                    if os.path.exists(fname):
                        pts = plyReadParticles(fname)
                        order = np.fromfile(fname + ".order", np.uint32)
                        order = np.argsort(order)
                        for j in range(3):
                            pts[order, j].tofile(fpath + dataset + "/" + str(s) + "-" + str(j) + ".draco." + str(qp) + ".out")
                        os.remove(fname)
                        os.remove(fname + ".order")
            print(i, qp)
