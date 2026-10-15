"""A single native LIBERO environment with explicit reset and timeout semantics."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from embodiedlab.interfaces import Action, Observation

UPSTREAM_COMMIT = '8561c60eea2fb93096146f240194649df73d8b1e'
ASSET_REVISION = '0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
# Current LeRobot profile, not OpenPI's different spatial horizon.
SUITE_STEPS = {'libero_spatial':280, 'libero_object':280, 'libero_goal':300,
               'libero_10':520, 'libero_90':400}


def _load_suite(name: str):
    """Configure upstream asset paths locally before its first, interactive import."""
    if name not in SUITE_STEPS:
        raise ValueError(f'Unknown suite {name!r}; choose {tuple(SUITE_STEPS)}')
    spec = importlib.util.find_spec('libero')
    if spec is None:
        raise ImportError('LIBERO is not installed; use the .venv-libero environment and README installation steps')
    package = Path(next(iter(spec.submodule_search_locations)))/'libero'
    config_dir = Path(os.environ.get('LIBERO_CONFIG_PATH',Path(__file__).resolve().parents[3]/'.libero'))
    config_dir.mkdir(parents=True,exist_ok=True)
    config_file = config_dir/'config.yaml'
    if not config_file.exists():
        # JSON is valid YAML, so no second configuration parser is required here.
        paths = {key:str(package/folder) for key,folder in
                 [('benchmark_root','.'),('bddl_files','bddl_files'),('init_states','init_files'),('assets','assets')]}
        paths['datasets'] = str(Path(__file__).resolve().parents[3]/'data/libero')
        config_file.write_text(json.dumps(paths,indent=2)+'\n')
    loaded = sys.modules.get('libero.libero')
    if loaded is not None and Path(loaded.config_file).resolve() != config_file.resolve():
        raise RuntimeError('LIBERO was already imported with another config; set LIBERO_CONFIG_PATH before import')
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    # Keep renderer/JIT caches writable without touching the user's home config.
    os.environ.setdefault('MPLCONFIGDIR',str(config_dir/'cache/matplotlib'))
    os.environ.setdefault('NUMBA_CACHE_DIR',str(config_dir/'cache/numba'))
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('PYOPENGL_PLATFORM',os.environ['MUJOCO_GL'])
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[name](task_order_index=0)


def list_task_ids(suite: str) -> list[int]:
    """List native task indices without constructing a simulation or renderer."""
    return list(range(_load_suite(suite).n_tasks))


def list_tasks(suite: str) -> list[dict]:
    """Return official task IDs, names and language without creating a simulator."""
    native_suite = _load_suite(suite)
    return [{'suite': suite, 'task_id': index, 'task_name': native_suite.get_task(index).name,
             'language': native_suite.get_task(index).language}
            for index in range(native_suite.n_tasks)]


class LiberoEnv:
    """Panda OSC_POSE: normalized world-frame pose deltas plus gripper, shape (7,).

    Action[:3] scales to ±0.05 m, action[3:6] to ±0.5 rad axis-angle;
    the rotation increment left-multiplies world attitude. Gripper -1 opens,
    +1 closes. These are not seven joint angles. Upstream owns all control,
    dynamics, reward and task-success logic.
    """

    def __init__(self, *, suite: str='libero_spatial', task_id: int=0, image_size: int=128,
                 max_episode_steps: int | None=None, init_state_index: int=0,
                 settle_steps: int=10) -> None:
        native_suite = _load_suite(suite)
        if type(task_id) is not int or not 0 <= task_id < native_suite.n_tasks:
            raise ValueError(f'task_id must be in [0,{native_suite.n_tasks-1}]')
        for key,value in [('image_size',image_size),('settle_steps',settle_steps)]:
            if type(value) is not int or value < (1 if key=='image_size' else 0):
                raise ValueError(f'{key} must be a valid nonnegative integer')
        limit = SUITE_STEPS[suite] if max_episode_steps is None else max_episode_steps
        if type(limit) is not int or limit < 1:
            raise ValueError('max_episode_steps must be a positive integer')
        self.suite,self.task_id,self.image_size = suite,task_id,image_size
        self.max_episode_steps,self.settle_steps = limit,settle_steps
        task = native_suite.get_task(task_id)
        self.task_name,self.language = task.name,task.language
        self._initial_states = np.asarray(native_suite.get_task_init_states(task_id))
        self.initial_state_count = len(self._initial_states)
        self.init_state_index = self._validate_index(init_state_index)
        self._task = task
        self._env = None
        self._ended = True
        self._closed = False
        self.steps = 0
        self.seed = None
        self.action_low,self.action_high = np.full(7,-1.),np.full(7,1.)

    def _validate_index(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < self.initial_state_count:
            raise ValueError(f'init_state_index must be in [0,{self.initial_state_count-1}]')
        return index

    def _create_env(self) -> None:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        bddl = Path(get_libero_path('bddl_files'))/self._task.problem_folder/self._task.bddl_file
        self._env = OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=self.image_size,
                                      camera_widths=self.image_size,control_freq=20,hard_reset=True,
                                      # Native done returns success, but robosuite internally
                                      # ends at horizon. Include settling to avoid premature end.
                                      horizon=self.max_episode_steps+self.settle_steps)
        controller = self._env.robots[0].controller
        if (not controller.use_delta or controller.control_dim != 6
                or not np.allclose(controller.input_min,-1.) or not np.allclose(controller.input_max,1.)
                or not np.allclose(controller.output_max,[.05,.05,.05,.5,.5,.5])
                or not np.allclose(controller.output_min,[-.05,-.05,-.05,-.5,-.5,-.5])):
            self.close()
            raise RuntimeError('Native LIBERO controller differs from the documented OSC delta contract')

    def reset(self, *, seed: int | None=None, init_state_index: int | None=None) -> tuple[Observation,dict]:
        """Hard reset, load an explicit official initial state, and settle 10 steps."""
        if self._closed:
            raise RuntimeError('Environment is closed; create a new LiberoEnv')
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('seed must be a nonnegative integer')
        if init_state_index is not None:
            self.init_state_index = self._validate_index(init_state_index)
        self._ended = True  # A failed native reset must never permit stepping partial state.
        if self._env is None:
            self._create_env()
        self.seed = seed
        self._env.seed(seed)
        self._env.reset()
        raw = self._env.set_init_state(self._initial_states[self.init_state_index])
        # Official no-op settling action opens the gripper. Its time is recorded
        # separately and is not counted as a policy transition.
        for _ in range(self.settle_steps):
            raw,_,_,_ = self._env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
        self.steps = 0
        self._ended = False
        return self._observation(raw),self._info(bool(self._env.check_success()))

    def _observation(self, raw: dict) -> Observation:
        from robosuite.utils.transform_utils import quat2axisangle
        state = np.r_[raw['robot0_eef_pos'],quat2axisangle(raw['robot0_eef_quat'].copy()),raw['robot0_gripper_qpos']]
        return {'robot_state':state.astype(np.float64),
                'joint_positions':raw['robot0_joint_pos'].copy(),
                'agentview_image':raw['agentview_image'].copy(),
                'wrist_image':raw['robot0_eye_in_hand_image'].copy(),
                'language':self.language}

    def _info(self, success: bool) -> dict:
        return {'success':success,'steps':self.steps,'simulation_time':self.steps/20.,
                'native_simulation_time':float(self._env.sim.data.time),
                'init_state_index':self.init_state_index,'suite':self.suite,'task_id':self.task_id}

    def step(self, action: Action) -> tuple[Observation,float,bool,bool,dict]:
        """Pass normalized OSC action unchanged; native success terminates, limit truncates."""
        if self._env is None or self._ended or self._closed:
            raise RuntimeError('Call reset before stepping, and do not step a finished episode')
        value = np.asarray(action,dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError('LIBERO action must be seven finite values')
        if np.any(value < -1.) or np.any(value > 1.):
            raise ValueError('LIBERO normalized actions must be in [-1,1]')
        raw,reward,done,_ = self._env.step(value)
        self.steps += 1
        success = bool(self._env.check_success())
        if bool(done) != success:
            raise RuntimeError('Unexpected native LIBERO done semantics: expected task success')
        terminated = success
        truncated = self.steps >= self.max_episode_steps and not terminated
        self._ended = terminated or truncated
        return self._observation(raw),float(reward),terminated,truncated,self._info(success)

    def episode_metadata(self) -> dict:
        """Public task/reset/control settings sufficient to rebuild a recorded run."""
        import importlib.metadata as metadata
        return {'env_name':'LiberoEnv','suite':self.suite,'task_id':self.task_id,
                'task_name':self.task_name,'language':self.language,'init_state_index':self.init_state_index,
                'image_size':self.image_size,'max_episode_steps':self.max_episode_steps,'settle_steps':self.settle_steps,
                'control_dt':.05,'control_mode':'osc_pose_delta_normalized','action_dim':7,
                'action_low':self.action_low.tolist(),'action_high':self.action_high.tolist(),
                'position_delta_scale_m':.05,'axis_angle_delta_scale_rad':.5,'gripper':'-1 open, +1 close',
                'image_orientation':'raw_robosuite_opengl','controller':'native Panda OSC_POSE',
                'evaluation_profile':'lerobot_single_env_hard_reset','libero_commit':UPSTREAM_COMMIT,
                'libero_version':metadata.version('hf-libero'),'assets_repo':'lerobot/libero-assets',
                'assets_revision':ASSET_REVISION,'assets_repo_type':'dataset','seed':self.seed,
                'mujoco_version':metadata.version('mujoco'),'robosuite_version':metadata.version('robosuite')}

    def close(self) -> None:
        """Release native simulation/renderer exactly once; repeated close is safe."""
        try:
            if self._env is not None:
                self._env.close()
        finally:
            self._env = None
            self._closed = True
            self._ended = True

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
