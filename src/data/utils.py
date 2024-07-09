import numpy as np
import SimpleITK as sitk

def resample_image(
        itk_image,
        new_size=None,
        out_spacing=None,
        interpolator="BSpline"  # "BSpline", "NearestNeighbor" is_label=False
) -> sitk.Image:
    # https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor

    if out_spacing is None:
        out_spacing = [
            sz * spc / nsz for nsz, sz, spc in
            zip(new_size, itk_image.GetSize(), itk_image.GetSpacing())
        ]

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_size = [
        int(np.round(original_size[i] * (original_spacing[i] / out_spacing[i]))) for i in range(len(original_size))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    interpolator = getattr(sitk, f"sitk{interpolator}")
    resample.SetInterpolator(interpolator)

    return resample.Execute(itk_image)