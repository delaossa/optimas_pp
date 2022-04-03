from collections import OrderedDict

# List the names of the varying parameters, and their range of value
# The names must be the same as those used in 'template_waket_script.py'
varying_parameters = OrderedDict({
    # 'intensity': [-0.1, 0.1],
    # 'tau': [-0.6, -0.1],
    # 'plasma_density': [0.0, 0.2],
    'beam_i0': [0.1, 7.5],  # kA
    'beam_i1': [0.1, 7.5]   # kA
    # 'beam_length': [-0.5, 0.]
    # 'beam_z': [-0.10, 0.10]
})
