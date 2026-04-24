# set base image (host OS)
FROM ghcr.io/tomographicimaging/cil:25.0.0

# EGL
RUN sudo apt-get update && apt-get install -y \
    libegl1-mesa \
    libnvidia-gl-550-server \
    openscad \
    mesa-utils \
    mesa-utils-extra \
    xterm

RUN sudo apt-get clean

# Change the workdir
WORKDIR /gvxr

# Install Python packages
RUN python3 -m pip install --user backports.tarfile
RUN python3 -m pip install --user gVXR viewscad SimpleITK progressbar k3d

# For debugging
# CMD xterm
