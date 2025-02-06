# Why are the distances given by the RealSense SO BAD???
* `[ ]` Inaccurate transform to camera, "ze_cam_xform_cal.py"
    - 
* `[ ]` Inaccurate camera intrinsics
* `[Y]` Bad Calibration, 2024-12-XX: Marginal improvment.
* `[Y]` [I would also recommend checking that the Threshold Filter option in the Post-Processing section of the Viewer's stereo module options is not enabled in order to ensure that the depth image is rendering the full distance of detail that it is able to observe instead of being limited](https://github.com/IntelRealSense/librealsense/issues/8258#issuecomment-768931256), 2025-02-06: No improvement
    1. Install viewer: `sudo apt-get install librealsense2-utils`
    1. Run: `realsense-viewer`
        - `[>]` Viewer recommends a firmware update 
            1. `rs-fw-update -l` to launch the tool and print a list of connected devices.
                * Intel RealSense D405 s/n 126122270157, update serial number: 125423070233, firmware version: 5.15.1
            1. Download: https://dev.intelrealsense.com/docs/firmware-releases-d400
            1. Unzip and navigate to directory with the unzipped BIN file
            1. `rs-fw-update -s 125423070233 -f Signed_Image_UVC_5_16_0_1.bin`, Replace file name with current
            1. `rs-fw-update -l`, Confirm update
    - 2025-02-06: Setting turned off, but there is no difference in the viewer
    - 2025-02-06: Tried other adjustments via the sliders, no performance improvement in the viewer

# Alternate Paths
* Match known cube to the image using edge detection
    - This KILLS open vocab