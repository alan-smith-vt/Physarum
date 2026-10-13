"""STEP -> binary STL (coarse), printing the bounding box.  usage: step2stl.py in.step out.stl [deflection_mm]"""
import sys
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

src, dst = sys.argv[1], sys.argv[2]
defl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
r = STEPControl_Reader()
assert r.ReadFile(src) == 1, "read failed"
r.TransferRoots()
shape = r.OneShape()
BRepMesh_IncrementalMesh(shape, defl, False, 0.5, True)
w = StlAPI_Writer()
w.ASCIIMode = False
assert w.Write(shape, dst)
b = Bnd_Box()
BRepBndLib.Add_s(shape, b)
print("bbox", [round(v, 2) for v in b.Get()])
