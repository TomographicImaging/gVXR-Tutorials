# set base image (host OS)
FROM ghcr.io/tomographicimaging/cil:25.0.0

USER root

RUN apt-get update && apt-get install -y \
    libegl1-mesa \
    libnvidia-gl-550 \
    mesa-utils \
    mesa-utils-extra \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean

USER "${NB_UID}"

# Change the workdir
WORKDIR /gvxr

COPY --chown=${NB_UID}:${NB_GID} environment.yml /tmp/
COPY --chown=${NB_UID}:${NB_GID} python/wheels/gvxr-2.1.0-cp311-cp311-manylinux_2_34_x86_64.whl /tmp/

RUN mamba create -f /tmp/environment.yml \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" && "/home/${NB_USER}"
