
project(
  'MCCNN', 
  'cpp',
  version: run_command('version.sh', 'get-vcs').stdout().strip(),
  default_options: [
    'cpp_std=c++17',
  ],
  meson_version: '>=1.1',
)

meson.add_dist_script('version.sh', 'set-dist', meson.project_version())

py = import('python').find_installation(pure: false)

pybind11_dep = dependency('pybind11')

install_subdir(
  'src/mc_cnn',
  install_dir: py.get_install_dir() / 'mc_cnn',
  strip_directory: true,
)


subdir('src/mc_cnn/cost_volume_cpp')