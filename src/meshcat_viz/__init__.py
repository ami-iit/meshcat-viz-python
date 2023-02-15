from . import logging, meshcat
from .model import MeshcatModel
from .model_builder import MeshcatModelBuilder
from .world import MeshcatWorld


def is_editable() -> bool:
    """
    Check if the package installation is either normal or editable.

    An editable installation is performed by either passing -e or --editable to pip.

    Returns:
        True if the installation is editable, False otherwise.
    """

    import importlib.util
    import pathlib
    import site

    # Get the ModuleSpec of the package
    package_spec = importlib.util.find_spec(name="meshcat_viz")

    # This can be None. If it's None, assume non-editable installation.
    if package_spec.origin is None:
        return False

    # Get the folder containing the package
    package_dir = str(pathlib.Path(package_spec.origin).parent.parent)

    # The installation is editable if the package dir is not in any {site|dist}-packages
    return package_dir not in site.getsitepackages()


# Initialize the logging verbosity
logging.configure(
    level=logging.LoggingLevel.DEBUG if is_editable() else logging.LoggingLevel.WARNING
)

del is_editable
