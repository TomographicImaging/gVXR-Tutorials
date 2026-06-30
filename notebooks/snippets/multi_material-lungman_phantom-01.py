gvxr.createOpenGLContext()

# or 
# backend = "OPENGL"
# backend = "EGL"
# gvxr.createNewContext(backend)
#
# with backend a string. Two backends are currently available:
#
#     "OPENGL": makes use of the windowing ability of your system. It can be used for realtime visualisations. 
#               It is available on Windows, GNU/Linux and MacOS computers.
#     "EGL": is for offscreen rendering on GNU/Linux computers. That's the option for cloud instances and supercomputers.
#     "OPENMP": implements a ray-tracer for CPU-only systems. It makes use of OpenMP to leverage parallelisation on multi-core CPUs.
