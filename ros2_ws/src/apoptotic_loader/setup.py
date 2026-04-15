from setuptools import setup
import os
from glob import glob
package_name = "apoptotic_loader"
setup(
    name=package_name, version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),  glob("config/*.yaml")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="Craig McClurkin", maintainer_email="craig.mcclurkin@louisville.edu",
    description="Apoptotic Model Loading framework for ROS 2",
    license="Apache-2.0", tests_require=["pytest"],
    entry_points={"console_scripts": [
        "apoptotic_manager    = apoptotic_loader.apoptotic_manager:main",
        "drift_observer       = apoptotic_loader.drift_observer:main",
        "checkpoint_registry  = apoptotic_loader.checkpoint_registry:main",
        "safe_stop_controller = apoptotic_loader.safe_stop_controller:main",
    ]},
)
