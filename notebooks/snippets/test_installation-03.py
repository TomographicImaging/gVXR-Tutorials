import os

if os.name == 'posix':
    gvxr.createNewContext("EGL") # For Linux
else:
    gvxr.createNewContext("OPENGL") # For Windows
