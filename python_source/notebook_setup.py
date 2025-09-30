# ipython magic
%matplotlib widget
%load_ext autoreload
%autoreload 2
import os, sys, inspect
# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.join(os.getcwd(), "python_source")
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)


%run python_source/matplotlib_header


