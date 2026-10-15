"""Isolated expert replay diagnostics; never changes the production reset protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np

from embodiedlab.envs.libero_env import LiberoEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5'


def load_episode(path: Path, name: str) -> dict:
    """Read small numeric references and original XML; camera arrays remain unread."""
    with h5py.File(path, 'r') as file:
        group = file[f'data/{name}']
        return {'name': name, 'actions': group['actions'][()], 'states': group['states'][()],
                'joints': group['obs/joint_states'][()], 'gripper': group['obs/gripper_states'][()],
                'eef': group['obs/ee_states'][()], 'xml': group.attrs['model_file']}


def relocate_assets(xml: str) -> str:
    """Change only source machine filenames, retaining every model parameter."""
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml
    tree = ET.fromstring(postprocess_model_xml(xml, {}))
    for node in tree.iter():
        filename = node.get('file')
        if filename and '/assets/' in filename and not Path(filename).is_file():
            node.set('file', str(Path(get_libero_path('assets')) / filename.split('/assets/', 1)[1]))
    return ET.tostring(tree, encoding='unicode')


def make_env(*, seed: int = 0, init_state_index: int = 0, settle_steps: int = 0,
             xml: str | None = None) -> LiberoEnv:
    """Create a native diagnostic environment; XML restore is isolated here only."""
    env = LiberoEnv(suite='libero_goal', task_id=7, image_size=32,
                    max_episode_steps=300, settle_steps=settle_steps)
    try:
        env.reset(seed=seed, init_state_index=init_state_index)
        if xml is not None:
            env._env.reset_from_xml_string(relocate_assets(xml))
        return env
    except Exception:
        env.close()
        raise


def controller_snapshot(native) -> dict:
    """Log unrecorded controller state without refreshing or changing its cache."""
    robot = native.robots[0]
    ctrl = robot.controller
    return {'new_update': bool(ctrl.new_update), 'goal_pos': ctrl.goal_pos.tolist(),
            'goal_ori': ctrl.goal_ori.tolist(), 'cached_ee_pos': ctrl.ee_pos.tolist(),
            'cached_ee_ori': ctrl.ee_ori_mat.tolist(), 'cached_joint_pos': ctrl.joint_pos.tolist(),
            'initial_joint': ctrl.initial_joint.tolist(),
            'gripper_current_action': robot.gripper.current_action.tolist(),
            'ctrl': native.sim.data.ctrl.tolist(), 'time': float(native.sim.data.time)}


def restore_physics(native, state: np.ndarray, *, sync_cache: bool = False,
                    reset_goal: bool = False, clear_physics: bool = False) -> dict:
    """Restore time/qpos/qvel once; every option is a labelled diagnostic factor.

    No variant is a complete original-state restore: control targets, nullspace
    history, gripper accumulator, warm-start and omitted early actions are absent.
    """
    if clear_physics:
        native.sim.reset()
    raw = native.set_init_state(state)
    if sync_cache or reset_goal:
        native.robots[0].controller.update(force=True)
    if reset_goal:
        native.robots[0].controller.reset_goal()
    return raw


def orientation_error(actual: np.ndarray, expected_axisangle: np.ndarray) -> float:
    """SO(3) geodesic error in radians, invariant to axis-angle branch choices."""
    from robosuite.utils.transform_utils import axisangle2quat, quat2mat
    expected = quat2mat(axisangle2quat(expected_axisangle.copy()))
    return float(np.arccos(np.clip((np.trace(expected.T @ actual) - 1.) / 2., -1., 1.)))


def contact_snapshot(native) -> dict:
    """Report endpoint contacts and forces; these do not cover the whole control interval."""
    model, data = native.sim.model, native.sim.data
    pairs = []
    for index, contact in enumerate(data.contact[:data.ncon]):
        names = [model.geom_id2name(contact.geom1) or str(contact.geom1),
                 model.geom_id2name(contact.geom2) or str(contact.geom2)]
        if any('stove' in name for name in names) and any('robot' in name or 'gripper' in name for name in names):
            force = np.zeros(6)
            mujoco.mj_contactForce(model._model, data._data, index, force)
            pairs.append({'geoms': names, 'normal_force_N': float(abs(force[0]))})
    return {'ncon': int(data.ncon), 'robot_stove_pairs': pairs,
            'robot_stove_normal_force_N': sum(pair['normal_force_N'] for pair in pairs)}


def object_snapshot(native, reference_state: np.ndarray | None = None) -> dict:
    """Compare nonrobot joints by name/type; fixed body positions are not in qpos."""
    model, data = native.sim.model, native.sim.data
    reference = None if reference_state is None else reference_state[1:1 + model.nq]
    result = {}
    for joint in range(model.njnt):
        name = model.joint_id2name(joint)
        if name.startswith('robot') or name.startswith('gripper'):
            continue
        kind, start = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
        width = {0: 7, 1: 4, 2: 1, 3: 1}[kind]
        value = data.qpos[start:start + width].copy()
        entry = {'joint_type': kind, 'qpos': value.tolist()}
        if reference is not None:
            target = reference[start:start + width]
            if kind == 0:
                entry['position_error_m'] = float(np.linalg.norm(value[:3] - target[:3]))
                dot = abs(float(np.dot(value[3:], target[3:]))) / (np.linalg.norm(value[3:]) * np.linalg.norm(target[3:]))
                entry['orientation_error_rad'] = float(2 * np.arccos(np.clip(dot, 0., 1.)))
            elif kind in (2, 3):
                entry['error_m' if kind == 2 else 'error_rad'] = float(abs(value[0] - target[0]))
        result[name] = entry
    return result


def step_metrics(native, raw: dict, source: dict, index: int) -> dict:
    """Compare native returned observation to stored post-action reference."""
    from robosuite.utils.transform_utils import quat2mat
    delta = raw['robot0_joint_pos'] - source['joints'][index]
    gripper = raw['robot0_gripper_qpos'] - source['gripper'][index]
    next_state = source['states'][index + 1] if index + 1 < len(source['states']) else None
    return {'index': index, 'source_pre_time_s': float(source['states'][index, 0]),
            'native_time_s': float(native.sim.data.time),
            'joint_errors_rad': delta.tolist(), 'joint_max_rad': float(np.max(np.abs(delta))),
            'joint_rmse_rad': float(np.sqrt(np.mean(delta ** 2))),
            'eef_position_error_m': float(np.linalg.norm(raw['robot0_eef_pos'] - source['eef'][index, :3])),
            'eef_orientation_error_rad': orientation_error(quat2mat(raw['robot0_eef_quat']), source['eef'][index, 3:]),
            'gripper_error_m': float(np.max(np.abs(gripper))),
            'objects': object_snapshot(native, next_state),
            'success': bool(native.check_success()), 'contacts': contact_snapshot(native),
            'zero_rotation_action': bool(np.all(source['actions'][index, 3:6] == 0.)),
            'gripper_action': float(source['actions'][index, 6])}


def replay(source: dict, plan: str, initial_index: int = 0) -> dict:
    """Execute expert actions with a single start reset, never overwrite later state."""
    xml = source['xml'] if plan == 'stored_xml_sync' else None
    settle = 10 if plan == 'official' else 0
    with make_env(init_state_index=initial_index, settle_steps=settle, xml=xml) as env:
        native = env._env
        restored_fixtures = []
        if plan == 'current_xml_sync':
            # A controlled XML round trip of this exact native scene: no hand
            # edits, no source geometry replacement or parameter fitting.
            native.reset_from_xml_string(native.sim.model.get_xml())
        if plan == 'fixtures_sync':
            # Restore only recorded fixed-scene initial positions, absent from
            # qpos. These values come directly from XML; no parameter is fitted.
            saved = mujoco.MjModel.from_xml_string(relocate_assets(source['xml']))
            current = native.sim.model._model
            mass_before, inertia_before = current.body_mass.copy(), current.body_inertia.copy()
            for name in ('flat_stove_1_main', 'wine_rack_1_main', 'wooden_cabinet_1_main'):
                dst = mujoco.mj_name2id(current, mujoco.mjtObj.mjOBJ_BODY, name)
                src = mujoco.mj_name2id(saved, mujoco.mjtObj.mjOBJ_BODY, name)
                assert dst >= 0 and src >= 0 and current.body_jntnum[dst] == 0
                restored_fixtures.append({'body': name, 'before_m': current.body_pos[dst].tolist(),
                                          'recorded_m': saved.body_pos[src].tolist()})
                current.body_pos[dst] = saved.body_pos[src]
            assert np.array_equal(current.body_mass, mass_before) and np.array_equal(current.body_inertia, inertia_before)
        if plan != 'official':
            restore_physics(native, source['states'][0], sync_cache=plan in ('history_sync', 'stored_xml_sync', 'current_xml_sync', 'fixtures_sync'))
        initial_controller = controller_snapshot(native)
        initial_qpos = native.sim.data.qpos.copy()
        ref_qpos = source['states'][0, 1:1 + native.sim.model.nq]
        rows = []
        first_success = None
        # Native success is checked each step. This diagnostic continues through
        # the recorded sequence, below the unchanged 300-step native horizon.
        for index, action in enumerate(source['actions']):
            raw, reward, done, _ = native.step(action)
            row = step_metrics(native, raw, source, index)
            assert bool(done) == row['success']
            row['reward'] = float(reward)
            rows.append(row)
            if row['success'] and first_success is None:
                first_success = index
        significant = next((row['index'] for row in rows if row['joint_max_rad'] > .05 or row['eef_position_error_m'] > .01), None)
        return {'episode': source['name'], 'plan': plan, 'executed_actions': len(rows),
                'source_actions': len(source['actions']), 'init_state_index': initial_index,
                'initial_qpos_max_error_mixed_units': float(np.max(np.abs(initial_qpos - ref_qpos))),
                'initial_joint_max_error_rad': float(np.max(np.abs(initial_qpos[:7] - ref_qpos[:7]))),
                'initial_controller': initial_controller, 'first_native_success_index': first_success,
                'recorded_fixture_positions_restored': restored_fixtures,
                'final_success': rows[-1]['success'], 'any_success': first_success is not None,
                'first_significant_index': significant,
                'significant_thresholds': {'joint_max_rad': .05, 'eef_position_error_m': .01},
                'partial_state_restore': plan != 'official',
                'initialization_matches_original': False,
                'rows': rows}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--episodes', nargs='+', default=['demo_0', 'demo_1', 'demo_2'])
    parser.add_argument('--plans', nargs='+', choices=['official', 'history_raw', 'history_sync', 'stored_xml_sync', 'current_xml_sync', 'fixtures_sync'],
                        default=['history_raw', 'history_sync'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; select a new filename')
    before = file_hash(args.dataset)
    results = []
    for name in args.episodes:
        source = load_episode(args.dataset, name)
        for plan in args.plans:
            result = replay(source, plan, int(name.split('_')[-1]))
            results.append(result)
            print(name, plan, 'actions=',result['executed_actions'], 'success=',result['final_success'],
                  'first_deviation=',result['first_significant_index'], flush=True)
    after = file_hash(args.dataset)
    assert before == after
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump({'scope': 'isolated partial-state diagnostics, not production evaluation',
                   'dataset_sha256_before': before, 'dataset_sha256_after': after,
                   'results': results}, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
