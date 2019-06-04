'''For Travis CI. Make our Linux and macOS wheels compatible with more machines!

ONLY RUN ONCE!
'''


from glob import glob
from os import remove
from os import replace
from subprocess import run
from sys import platform


WHEELS = glob('dist/*.whl')


if __name__ == '__main__':

    assert WHEELS, 'No wheels in dist!'

    print('\nBefore:', *WHEELS, sep='\n - ', end='\n\n')

    if platform == 'linux':

        # We're typically eligible for manylinux1... or at least manylinux2010.

        # This will remove the wheel if it was unchanged... but that will cause
        # our assert to fail later, which is what we want!

        for wheel in WHEELS:

            run(('auditwheel', 'repair', wheel, '-w', 'dist'), check=True)
            remove(wheel)

    elif platform == 'darwin':

        # We lie here, and say our 10.9 64-bit build is a 10.6 32/64-bit one.

        # This is because pip is conservative in what wheels it will use, but
        # Python installations are EXTREMELY liberal in their macOS support.
        # A typical user may be running a 32/64 Python built for 10.6.

        # In reality, we shouldn't worry about supporting 32-bit Snow Leopard.

        for wheel in WHEELS:

            fake = wheel.replace('macosx_10_9_x86_64', 'macosx_10_6_intel')
            replace(wheel, fake)

            assert wheel != fake, 'We expected a macOS 10.9 x86_64 build!'
    
    # Windows is fine.

    FIXED = glob('dist/*.whl')

    print('\nAfter:', *glob('dist/*.whl'), sep='\n - ', end='\n\n')

    assert len(WHEELS) == len(FIXED), 'We gained or lost a wheel!'
