import os
import sys

# Ensure Windows DLL search path includes torch\lib
if sys.platform == "win32":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    torch_lib = os.path.join(project_root, "venv", "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
