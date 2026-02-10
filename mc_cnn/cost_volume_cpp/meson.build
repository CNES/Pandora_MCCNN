cost_volume_includes = include_directories('include')
cost_volume_src = files('src/cv_pixelmajor_notorch.cpp')

bind_src = files('src/bind.cpp')


py.extension_module(
    'cost_volume_bind',
    [cost_volume_src, bind_src],
    subdir: 'mc_cnn/cost_volume_cpp',
    install: true,
    include_directories: [cost_volume_includes],
    dependencies: [pybind11_dep],
)