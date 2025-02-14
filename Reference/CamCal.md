# Camera Calibration Tool Installation
1. `sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo jammy main"`
1. `sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE`
1. `sudo apt update`
1. `sudo apt install librscalibrationtool`
1. `sudo apt install libusb-dev libusb-1.0-0-dev libglfw3 libglfw3-dev freeglut3 freeglut3-dev`

# Verify Installation && Run Dymaic Target Calibration
1. Print and flatly mount "print-target-fixed-width.pdf"
1. `/usr/bin/Intel.Realsense.DynamicCalibrator -v`
1. `/usr/bin/Intel.Realsense.DynamicCalibrator -list`
1. `/usr/bin/Intel.Realsense.CustomRW -r -sn 126122270157`
1. Point the camera in a convenient direction that allows you to move the target freely within the camera's FOV
1. `/usr/bin/Intel.Realsense.DynamicCalibrator`
1. Follow the onscreen instructions in order to calibrate the camera and upload new intrinsics.