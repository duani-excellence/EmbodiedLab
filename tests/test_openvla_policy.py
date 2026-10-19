"""OFT adapter unit fixtures; these arrays never represent model benchmark scores."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('openpi_client', reason='Optional shared websocket client is not installed')

from embodiedlab.policies import openvla_oft as adapter


@pytest.fixture
def observation():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:32, :64] = [255, 0, 0]
    image[-32:, -64:] = [0, 255, 0]
    wrist = np.zeros_like(image)
    wrist[:128, :] = [0, 0, 255]
    return {'agentview_image': image, 'wrist_image': wrist,
            'robot_state': np.arange(8, dtype=np.float64) / 10.,
            'language': 'turn on the stove and put the moka pot on it'}


class FixtureClient:
    """Explicit wire fixture with distinct pre/post gripper processing arrays."""

    def __init__(self):
        self.actions = np.arange(56, dtype=np.float32).reshape(8, 7) / 100.
        self.actions[:, 6] = [-1., 1.] * 4
        self.raw_chunk = self.actions.copy()
        self.raw_chunk[:, 6] = np.linspace(0., 1., 8)
        self.received, self.resets, self.closes = [], 0, 0

    def get_server_metadata(self):
        return {'fixture': 'not a loaded VLA'}

    def reset(self):
        self.resets += 1

    def infer(self, payload):
        self.received.append(deepcopy(payload))
        return {'actions': self.actions, 'raw_model_chunk': self.raw_chunk,
                'server_request_index': len(self.received) - 1,
                'synchronized_sample_ms': 1.25, 'policy_timing': {'infer_ms': 1.5},
                'preprocessing': {'fixture_only': True}}

    def close(self):
        self.closes += 1


def test_raw_images_state_prompt_and_suite_reach_server_without_rotation_or_crop(observation):
    before = deepcopy(observation)
    client = FixtureClient()
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    policy.act(observation)
    payload = client.received[0]
    assert set(payload) == {'agentview_image', 'wrist_image', 'robot_state', 'prompt', 'suite'}
    assert payload['suite'] == 'libero_10'
    assert payload['prompt'] == before['language']
    for key in ('agentview_image', 'wrist_image'):
        np.testing.assert_array_equal(payload[key], before[key])
        np.testing.assert_array_equal(observation[key], before[key])
        assert payload[key].shape == (256, 256, 3)
        assert payload[key].dtype == np.uint8 and payload[key].flags.c_contiguous
        assert policy.request_log[0]['raw_image_sha256'][key] == hashlib.sha256(before[key].tobytes()).hexdigest()
    assert tuple(payload['agentview_image'][0, 0]) == (255, 0, 0)
    assert tuple(payload['agentview_image'][-1, -1]) == (0, 255, 0)
    assert tuple(payload['wrist_image'][0, 0]) == (0, 0, 255)
    assert tuple(payload['wrist_image'][-1, 0]) == (0, 0, 0)
    np.testing.assert_array_equal(payload['robot_state'], before['robot_state'])
    assert payload['robot_state'].dtype == np.float64
    payload['robot_state'][:] = -99.
    np.testing.assert_array_equal(observation['robot_state'], before['robot_state'])


def test_eight_actions_execute_in_order_then_ninth_uses_new_observation(observation):
    client = FixtureClient()
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    assert policy.server_metadata == {'fixture': 'not a loaded VLA'}
    assert policy.replan_steps == 8
    for index in range(8):
        action = policy.act(observation)
        np.testing.assert_array_equal(action, client.actions[index])
        assert action.shape == (7,) and action.dtype == np.float64
        action[:] = 0.  # Returning one action cannot mutate other cached actions.
    assert len(client.received) == 1
    observation['robot_state'] += .1
    np.testing.assert_array_equal(policy.act(observation), client.actions[0])
    assert len(client.received) == 2
    np.testing.assert_array_equal(client.received[1]['robot_state'], observation['robot_state'])
    assert [row['request_step'] for row in policy.request_log] == [0, 8]
    assert [row['server_request_index'] for row in policy.request_log] == [0, 1]
    assert len(policy.act_times_ms) == 9 and all(ms >= 0. for ms in policy.act_times_ms)


def test_reset_discards_buffer_and_episode_logs_without_assuming_remote_rng_reset(observation):
    client = FixtureClient()
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    policy.act(observation)
    assert len(policy._actions) == 7
    policy.reset()
    assert not policy._actions and not policy.request_log and not policy.act_times_ms
    assert policy.projected_action_count == 0 and policy.projected_dimension_counts == [0] * 7
    assert policy.projection_log == [] and policy.max_action_overshoot == 0.
    np.testing.assert_array_equal(policy.act(observation), client.actions[0])
    assert len(client.received) == 2 and client.resets == 2
    assert policy.request_log[0]['request_step'] == 0


def test_raw_and_final_chunks_are_distinct_logged_without_second_gripper_transform(observation):
    client = FixtureClient()
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    issued = np.stack([policy.act(observation) for _ in range(8)])
    np.testing.assert_array_equal(issued, client.actions)
    record = policy.request_log[0]
    np.testing.assert_array_equal(record['raw_chunk'], client.raw_chunk)
    np.testing.assert_array_equal(record['issued_chunk'], client.actions)
    assert not np.array_equal(record['raw_chunk'], record['issued_chunk'])
    assert record['synchronized_sample_ms'] == 1.25
    assert record['policy_timing'] == {'infer_ms': 1.5}
    assert record['server_preprocessing'] == {'fixture_only': True}
    assert record['rpc_ms'] >= 0.
    assert policy.projected_action_count == 0 and not policy.projection_log


@pytest.mark.parametrize('key,value', [
    ('agentview_image', np.zeros((128, 128, 3), dtype=np.uint8)),
    ('wrist_image', np.zeros((256, 256, 3), dtype=np.float32)),
    ('robot_state', np.zeros(7)), ('robot_state', np.full(8, np.nan)),
    ('robot_state', np.full(8, np.inf)), ('language', ''), ('language', 7),
])
def test_invalid_inputs_fail_before_any_request(observation, key, value):
    client = FixtureClient()
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    observation[key] = value
    with pytest.raises(ValueError):
        policy.act(observation)
    assert not client.received and not policy.request_log and not policy._actions
    assert len(policy.act_times_ms) == 1


@pytest.mark.parametrize('field,value', [
    ('actions', np.zeros((7, 7))), ('actions', np.zeros((8, 8))),
    ('actions', np.full((8, 7), np.nan)), ('actions', np.full((8, 7), np.inf)),
    ('actions', np.full((8, 7), 1.00000001)), ('actions', np.full((8, 7), -1.01)),
    ('raw_chunk', np.zeros((7, 7))), ('raw_chunk', np.zeros((8, 8))),
    ('raw_chunk', np.full((8, 7), np.nan)), ('raw_chunk', np.full((8, 7), np.inf)),
])
def test_invalid_model_chunks_fail_without_clipping_and_preserve_reply(observation, field, value):
    client = FixtureClient()
    setattr(client, field, value)
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    with pytest.raises(ValueError, match='finite|shape|outside'):
        policy.act(observation)
    record = policy.request_log[0]
    np.testing.assert_array_equal(record['raw_chunk'], client.raw_chunk)
    np.testing.assert_array_equal(record['issued_chunk'], client.actions)
    assert not policy._actions and policy.projected_action_count == 0
    assert not policy.projection_log and len(policy.act_times_ms) == 1


def test_eighth_action_is_validated_before_first_action_is_issued(observation):
    client = FixtureClient()
    client.actions[7, 2] = 2.
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    with pytest.raises(ValueError, match='outside'):
        policy.act(observation)
    assert not policy._actions
    assert policy.request_log[0]['issued_chunk'][7][2] == 2.


def test_rpc_failure_is_recorded_and_close_is_idempotent(observation):
    client = FixtureClient()
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    def fail(_payload):
        raise TimeoutError('explicit fixture receive deadline')
    client.infer = fail
    with pytest.raises(TimeoutError, match='deadline'):
        policy.act(observation)
    assert 'TimeoutError' in policy.request_log[0]['error']
    assert policy.request_log[0]['rpc_ms'] >= 0. and not policy._actions
    policy.close()
    policy.close()
    assert client.closes == 1
    with pytest.raises(RuntimeError, match='closed'):
        policy.act(observation)
    with pytest.raises(RuntimeError, match='closed'):
        policy.reset()


def test_connection_failure_propagates_with_one_bounded_attempt(monkeypatch):
    calls = []
    def refuse(*args):
        calls.append(args)
        raise ConnectionRefusedError('explicit fixture connection refusal')
    monkeypatch.setattr(adapter, '_BoundedClient', refuse)
    with pytest.raises(ConnectionRefusedError, match='refusal'):
        adapter.OpenVLAOFTPolicy(suite='libero_10', connect_timeout=3., request_timeout=7.)
    assert calls == [('127.0.0.1', 8007, 3., 7.)]


def test_close_marks_policy_closed_even_when_transport_close_fails():
    client = FixtureClient()
    def fail_close():
        client.closes += 1
        raise OSError('explicit fixture close failure')
    client.close = fail_close
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=client)
    with pytest.raises(OSError, match='close failure'):
        policy.close()
    policy.close()
    assert client.closes == 1
    with pytest.raises(RuntimeError, match='closed'):
        policy.reset()


def environment_metadata():
    return {'suite': 'libero_10', 'image_size': 256, 'action_dim': 7,
            'image_orientation': 'raw_robosuite_opengl', 'settle_steps': 10,
            'control_mode': 'osc_pose_delta_normalized', 'control_dt': .05,
            'position_delta_scale_m': .05, 'axis_angle_delta_scale_rad': .5,
            'gripper': '-1 open, +1 close'}


def test_matching_environment_contract_is_accepted():
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=FixtureClient())
    policy.assert_compatible(SimpleNamespace(episode_metadata=environment_metadata))


@pytest.mark.parametrize('key,value', [('suite', 'libero_spatial'), ('image_size', 128),
                                     ('action_dim', 6), ('image_orientation', 'rotated'),
                                     ('settle_steps', 0), ('control_mode', 'joint_position'),
                                     ('control_dt', .1), ('position_delta_scale_m', .1),
                                     ('axis_angle_delta_scale_rad', 1.), ('gripper', '+1 open')])
def test_wrong_suite_or_environment_contract_is_rejected(key, value):
    metadata = environment_metadata()
    metadata[key] = value
    policy = adapter.OpenVLAOFTPolicy(suite='libero_10', client=FixtureClient())
    with pytest.raises(ValueError, match=key):
        policy.assert_compatible(SimpleNamespace(episode_metadata=lambda: metadata))


@pytest.mark.parametrize('kwargs', [{'suite': 'libero_90'}, {'suite': 'unknown'},
                                    {'connect_timeout': 0}, {'request_timeout': float('inf')},
                                    {'request_timeout': float('nan')}])
def test_invalid_configuration_is_rejected_before_connection(kwargs):
    arguments = {'suite': 'libero_10', 'client': FixtureClient(), **kwargs}
    with pytest.raises(ValueError):
        adapter.OpenVLAOFTPolicy(**arguments)
