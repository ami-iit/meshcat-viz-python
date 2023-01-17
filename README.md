# meshcat-viz-python

**Generic visualizer of robot models based on MeshCat.**

## Installation

You can install the project with [`pypa/pip`][pip], preferably in a [virtual environment][venv]:

```bash
pip install git+https://github.com/ami-iit/meshcat-viz-python
```

[pip]: https://github.com/pypa/pip/
[venv]: https://docs.python.org/3.8/tutorial/venv.html

## Example

<details>
<summary>Visualize a manipulator</summary>

```python
import pathlib

import gym_ignition_models
import numpy as np

from meshcat_viz.world import MeshcatWorld

# Load a model resource
model_sdf_path = pathlib.Path(
    gym_ignition_models.get_model_resource(
        robot_name="panda", resource_type=gym_ignition_models.ResourceType.SDF_PATH
    )
)

# Open the visualizer
world = MeshcatWorld()
world.open()

# Insert the model from a URDF/SDF resource.
# Note: for URDF files support, check details in https://github.com/ami-iit/rod.
model_name = world.insert_model(model_description=model_sdf_path)

# Update the base position
world.update_model(model_name=model_name, base_position=np.array([1.0, 0, 0]))

# Update the joint positions
s = world._jaxsim_models[model_name].joint_random_positions()
world.update_model(model_name=model_name, joint_positions=s)
```

</details>

## Contributing

Pull requests are welcome. 
For major changes, please open an issue first to discuss what you would like to change.

## Maintainers

| [<img src="https://github.com/diegoferigo.png" width="40">][df] | [@diegoferigo][df] |
|:---------------------------------------------------------------:|:------------------:|

[df]: https://github.com/diegoferigo

## License

[BSD3](https://choosealicense.com/licenses/bsd-3-clause/)

