from setuptools import find_packages, setup

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='piyush',
    maintainer_email='piyush.sahoo456@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'straight_vel =navigation.simple_vel:main',
            'hue_detect =navigation.hue_depth:main',
            'nav=navigation.nav:main',
            'nav2=navigation.nav2:main',
            'nav3=navigation.nav3:main',
            'nav4=navigation.nav4:main',
            'nav5=navigation.nav5:main',
            'temp_match=navigation.hue_mask:main',
            'turn_cv2=navigation.turn:main',
            'turn_2=navigation.turn2:main',
            'turn_3=navigation.turn3:main',
            'turn_4=navigation.turn4:main',
            'turn_5=navigation.turn_5:main',
        ],
    },
)
