from distutils.core import setup,Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import eigency
import os

os.environ["CC"] = "/usr/bin/g++"
os.environ["CXX"] = "/usr/bin/g++"


#packageDir = os.path.dirname(__file__)
#includedDir = [packageDir]
#os.chdir(packageDir)

#ext_modules = [
#Extension("predictc", ["predictc.pyx"], include_dirs=includedDir),
#]

setup(
	ext_modules=cythonize(Extension(
		name='predictc',
		author='anonymous',
		version='0.0.1',
		sources=['predictc.pyx'],
		language='c++',
		extra_compile_args=["-std=c++11","-O2",],
		include_dirs=[".", "module-dir-name"] + eigency.get_includes()+["./eigen3", "./libigl/include"],
		#If you have installed eigen, you can configure your own path. When numpy references errors, you need to import its header file
		install_requires=['Cython>=0.2.15','eigency>=1.77'],
		packages=['little-try'],
		python_requires='>=3',
		
		define_macros=[('CYTHON_COMPILE', 'true')]
		#define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	))
)