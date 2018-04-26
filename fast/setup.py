from distutils.core import setup, Extension
import numpy

opencv_sources = ['KCF.cpp', 'TrackerCustomKCF.cpp']
kcf = Extension('KCF', sources=opencv_sources,
                 include_dirs=[numpy.get_include()],
                 libraries=['opencv_core', 'opencv_tracking', 'opencv_highgui', 'opencv_videoio'],
                 language='c++',
                 extra_compile_args=['-std=c++11', '-arch=x86_64'],
                 )

opencv_sources = ['TLD.cpp']
tld = Extension('TLD', sources=opencv_sources,
                 include_dirs=[numpy.get_include()],
                 libraries=['opencv_core', 'opencv_tracking', 'opencv_highgui', 'opencv_videoio'],
                 language='c++',
                 extra_compile_args=['-std=c++11'],
                 )

# run the setup
setup(ext_modules=[tld])
