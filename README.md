# Iterative Closest Point

A Python implementation of the [Iterative closest point][1] algorithm for 2D point clouds, based on the paper
"Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans" by F. Lu and E. Milios.

## Requirements
The required packages can be installed by executing:
```
python -m pip install -r requirements.txt
```

## Example
An example of usage is given in the `examples` directory.

Note that the example is assuming that the directory containing the provided module (i.e. `icp.py`) is on `sys.path`.


[1]: https://en.wikipedia.org/wiki/Iterative_closest_point
