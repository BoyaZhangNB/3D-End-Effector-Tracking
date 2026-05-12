import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path(
    "franka_emika_panda/scene.xml"
)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = -3 * np.ones_like([1] * model.nu)
        mujoco.mj_step(model, data)
        time.sleep(0.2)
        print(data.qpos)
        viewer.sync()