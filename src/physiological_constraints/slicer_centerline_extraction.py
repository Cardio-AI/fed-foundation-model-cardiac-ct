
from ExtractCenterline import ExtractCenterlineLogic
import slicer
import vtk
import numpy as np
import SimpleITK as sitk

def mps_from_pts(pts, path):
    xml_from_pt = lambda i, pt: f"""<point>
                <id>{i}</id>
                <specification>0</specification>
                <x>{pt[0]}</x>
                <y>{pt[1]}</y>
                <z>{pt[2]}</z>
            </point>
            """
    mps_string = f"""<?xml version="1.0" encoding="UTF-8"?>
<point_set_file>
    <file_version>0.1</file_version>
    <point_set>
        <time_series>
            <time_series_id>0</time_series_id>
            <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                <IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0" m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0" m_2_1="0" m_2_2="1"/>
                <Offset type="Vector3D" x="0" y="0" z="0"/>
                <Bounds>
                    <Min type="Vector3D" x="{np.min(pts[:,0])}" y="{np.min(pts[:,1])}" z="{np.min(pts[:,2])}"/>
                    <Max type="Vector3D" x="{np.max(pts[:,0])}" y="{np.max(pts[:,1])}" z="{np.max(pts[:,2])}"/>
                </Bounds>
            </Geometry3D>
            {''.join([xml_from_pt(i, pt) for i, pt in enumerate(pts)])}
        </time_series>
    </point_set>
</point_set_file>"""
    with open(path, 'w') as f:
        f.write(mps_string)

def extract_centerline(src, dst, start_point, origin):
    if start_point is not None:
        print('LOADING STARTING POINT')
        with open(start_point, 'rb') as f:
            start_point = np.load(f)
    
    with open(origin, 'rb') as f:
        origin = np.load(f)

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(src)
    reader.Update()
    sitk_image = sitk.ReadImage(src)
        
    threshold_value = 0.5
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(reader.GetOutputPort())  # Use the output from the reader or your segmentation filter
    marching_cubes.SetValue(0, threshold_value)  # Set the threshold value for your segmentation
    marching_cubes.Update()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(marching_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(100)  # Adjust the number of smoothing iterations
    smoother.SetRelaxationFactor(0.1)   # Adjust the relaxation factor

    decimator = vtk.vtkDecimatePro()
    decimator.SetInputConnection(smoother.GetOutputPort())
    decimator.SetTargetReduction(0.5)  # Adjust the target reduction factor

    decimator.Update()

    surface_mapper = vtk.vtkPolyDataMapper()
    surface_mapper.SetInputConnection(decimator.GetOutputPort())
    surfacePolyData = surface_mapper.GetInput()

    targetNumberOfPoints = 5000.
    decimationAggressiveness = 4.0
    subdivide = False

    logic = ExtractCenterlineLogic()
    preprocessedPolyData = logic.preprocess(surfacePolyData, targetNumberOfPoints, decimationAggressiveness, subdivide) # <- in default argumetns
    endPointsMarkupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
        slicer.mrmlScene.GetUniqueNameByString("Centerline endpoints"))

    # orig = np.array(sitk_image.GetOrigin())
    networkPolyData = logic.extractNetwork(preprocessedPolyData, endPointsMarkupsNode=None)
    endpointPositions = logic.getEndPoints(networkPolyData, startPointPosition=None)
    print(endpointPositions)
    if start_point is not None:
        endpointPositions = [
            start_point,
            endpointPositions[1]
            # (71.256, 74.162, 71.193),
            # (45.156, 86.829, 118.512)
        ]

    endPointsMarkupsNode.RemoveAllControlPoints()
    for position in endpointPositions:
        endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))
    centerlinePolyData, voronoiDiagramPolyData = logic.extractCenterline(surfacePolyData, endPointsMarkupsNode, curveSamplingDistance=1.0)
    vtk_points = centerlinePolyData.GetPoints()
    print(centerlinePolyData, vtk_points)
    # sitk_image.SetOrigin()
    points = []
    num_points = vtk_points.GetNumberOfPoints()
    for i in range(num_points):
        point = vtk_points.GetPoint(i)
        point = np.array(point) #+ origin
        points.append(point.tolist())
    points_idx = [sitk_image.TransformPhysicalPointToContinuousIndex(pt) for pt in points]
    sitk_image.SetOrigin(origin.tolist())
    points = [sitk_image.TransformContinuousIndexToPhysicalPoint(pt) for pt in points_idx]
    mps_from_pts(np.array(points), f'{dst}/centerline.mps')
    np.save(f'{dst}/centerline.npy', np.array(points))
    output_file_name = f'{dst}/centerline.obj'
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(output_file_name)
    writer.SetInputData(centerlinePolyData)
    writer.Write()
    import sys
    sys.exit()
    
if __name__ == '__main__':
    import sys
    extract_centerline(*sys.argv[1:])

# /mnt/ssd/applications/Slicer-5.2.2-linux-amd64/Slicer --no-main-window --python-script /mnt/ssd/git-repos/floto-tavi-outcome-prediction/slicer_centerline_extraction.py /mnt/ssd/data/tavi_cts/Aorta CT/D1/SEG/seg.nii.gz /mnt/ssd/data/tavi_cts/Aorta CT/D1/