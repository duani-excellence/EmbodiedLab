"""Adapter-only tests: fake server replies never count as real model results."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('openpi_client', reason='Optional OpenPI client is not installed')
from openpi_client import image_tools, msgpack_numpy


from embodiedlab.policies import openpi as adapter
OpenPIPolicy = adapter.OpenPIPolicy


@pytest.fixture
def observation() -> dict:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:64, :64] = [255, 0, 0]
    image[-64:, -64:] = [0, 255, 0]
    wrist = np.zeros_like(image)
    wrist[:, :128] = [0, 0, 255]
    return {'agentview_image': image, 'wrist_image': wrist,
            'robot_state': np.arange(8, dtype=np.float64) / 10,
            'language': 'open the middle drawer of the cabinet'}


class Client:
    """Wire fixture only; outputs are deterministic test arrays, never simulated expertise."""

    def __init__(self, chunk=None):
        self.chunk = np.arange(70, dtype=np.float32).reshape(10, 7) / 100 if chunk is None else chunk
        self.received, self.resets, self.closes = [], 0, 0

    def get_server_metadata(self):
        return {'test_fixture': True}

    def reset(self):
        self.resets += 1

    def infer(self, payload):
        self.received.append(deepcopy(payload))
        return {'actions': self.chunk, 'server_timing': {'infer_ms': 1.},
                'server_request_index': len(self.received) - 1, 'synchronized_sample_ms': .8}

    def close(self):
        self.closes += 1


def test_input_matches_official_rotation_resize_state_and_prompt_without_mutation(observation):
    before = deepcopy(observation)
    payload = adapter.prepare_openpi_observation(observation)
    for source, target in [('agentview_image', 'observation/image'), ('wrist_image', 'observation/wrist_image')]:
        expected = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(observation[source][::-1, ::-1]), 224, 224))
        np.testing.assert_array_equal(payload[target], expected)
        assert payload[target].shape == (224, 224, 3) and payload[target].dtype == np.uint8
        assert payload[target].flags.c_contiguous
        np.testing.assert_array_equal(observation[source], before[source])
    assert tuple(payload['observation/image'][0, 0]) == (0, 255, 0)
    assert tuple(payload['observation/wrist_image'][0, 0]) == (0, 0, 0)
    assert tuple(payload['observation/wrist_image'][0, -1]) == (0, 0, 255)
    np.testing.assert_array_equal(payload['observation/state'], before['robot_state'])
    assert payload['prompt'] == before['language']
    payload['observation/state'][:] = -100.
    np.testing.assert_array_equal(observation['robot_state'], before['robot_state'])


@pytest.mark.parametrize('key,value', [
    ('agentview_image', np.zeros((128, 128, 3), dtype=np.uint8)),
    ('wrist_image', np.zeros((256, 256, 3), dtype=np.float32)),
    ('robot_state', np.zeros(7)), ('robot_state', np.full(8, np.nan)), ('language', ''),
])
def test_bad_input_rejected_before_request(observation, key, value):
    client = Client()
    policy = OpenPIPolicy(client=client)
    observation[key] = value
    with pytest.raises(ValueError):
        policy.act(observation)
    assert client.received == [] and policy.request_log == []


def test_first_five_cached_rest_discarded_reset_and_logs(observation):
    client = Client()
    policy = OpenPIPolicy(client=client)
    assert policy.server_metadata == {'test_fixture': True}
    for index in range(5):
        action = policy.act(observation)
        np.testing.assert_array_equal(action, client.chunk[index])
        assert action.shape == (7,) and action.dtype == np.float64
    assert len(client.received) == 1
    np.testing.assert_array_equal(policy.act(observation), client.chunk[0])
    assert len(client.received) == 2
    assert [row['request_step'] for row in policy.request_log] == [0, 5]
    np.testing.assert_array_equal(policy.request_log[0]['raw_chunk'], client.chunk)
    assert all(row['rpc_ms'] >= 0. for row in policy.request_log)
    assert [row['server_request_index'] for row in policy.request_log] == [0, 1]
    assert all(row['synchronized_sample_ms'] == .8 for row in policy.request_log)
    assert len(policy.act_times_ms) == 6 and all(t >= 0. for t in policy.act_times_ms)
    policy.reset()
    assert not policy.request_log and not policy.act_times_ms and policy.last_raw_chunk is None
    assert not policy.projection_log and policy.projected_action_count == 0
    assert policy.projected_dimension_counts == [0] * 7 and policy.max_action_overshoot == 0.
    np.testing.assert_array_equal(policy.act(observation), client.chunk[0])
    assert len(client.received) == 3  # Buffered remainder was discarded.
    assert client.resets == 2  # Constructor and explicit reset; no remote RNG assertion.


@pytest.mark.parametrize('chunk', [np.zeros((9, 7)), np.zeros((10, 8)),
                                  np.full((10, 7), np.nan), np.full((10, 7), 1.01)])
def test_bad_model_output_keeps_raw_chunk_and_fails_without_clipping(observation, chunk):
    policy = OpenPIPolicy(client=Client(chunk))
    with pytest.raises(ValueError, match='shape|finite|outside'):
        policy.act(observation)
    np.testing.assert_array_equal(policy.last_raw_chunk, chunk)
    np.testing.assert_array_equal(policy.request_log[0]['raw_chunk'], chunk)
    assert 'error' in policy.request_log[0] and len(policy.act_times_ms) == 1
    assert not policy._actions


def test_strict_bounds_only_check_executed_prefix_but_finiteness_checks_whole_chunk(observation):
    chunk = np.zeros((10, 7))
    chunk[5:, :] = 2.
    policy = OpenPIPolicy(client=Client(chunk))
    for _ in range(5):
        np.testing.assert_array_equal(policy.act(observation), np.zeros(7))
    assert policy.projected_action_count == 0 and len(policy.request_log) == 1
    np.testing.assert_array_equal(policy.last_raw_chunk[5:], np.full((5, 7), 2.))
    chunk[9, 6] = np.nan
    policy = OpenPIPolicy(client=Client(chunk), native_action_bounds=True)
    with pytest.raises(ValueError, match='finite'):
        policy.act(observation)


def test_native_bounds_explicitly_project_without_rescaling_or_gripper_threshold(observation):
    chunk = np.zeros((10, 7))
    chunk[0] = [-1.2, 1.1, .3, -.4, .5, -.6, -1.002]
    chunk[1, 6] = .23
    chunk[2, 6] = 1.008
    policy = OpenPIPolicy(client=Client(chunk), native_action_bounds=True)
    for index in range(5):
        np.testing.assert_array_equal(policy.act(observation), np.clip(chunk[index], -1., 1.))
    np.testing.assert_array_equal(policy.last_raw_chunk, chunk)
    np.testing.assert_array_equal(policy.request_log[0]['raw_chunk'], chunk)
    assert policy.projection_log[0]['clipped_dims'] == [0, 1, 6]
    assert policy.projection_log[1]['issued_action'][6] == .23
    assert policy.projected_action_count == 2
    assert policy.projected_dimension_counts == [1, 1, 0, 0, 0, 0, 2]
    assert policy.max_action_overshoot == pytest.approx(.2)
    assert policy.projection_log[0]['maxovershoot'] == pytest.approx(.2)
    policy.reset()
    assert policy.projection_log == [] and policy.projected_action_count == 0


REFERENCE_CHUNK = Path(os.environ.get('EMBODIEDLAB_OPENPI_REFERENCE_CHUNK', ''))


@pytest.mark.skipif(not REFERENCE_CHUNK.is_file(), reason='Optional captured official-model chunk not supplied')
def test_captured_real_reference_chunk_preserves_first_five_action_order(observation):
    """Replay captured predictions through the adapter; this test does not run the model."""
    chunk = np.load(REFERENCE_CHUNK, allow_pickle=False)
    assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
    strict = OpenPIPolicy(client=Client(chunk))
    with pytest.raises(ValueError, match='outside'):
        strict.act(observation)
    np.testing.assert_array_equal(strict.last_raw_chunk, chunk)
    native = OpenPIPolicy(client=Client(chunk), native_action_bounds=True)
    actual = np.stack([native.act(observation) for _ in range(5)])
    np.testing.assert_array_equal(actual, np.clip(chunk[:5], -1., 1.))
    np.testing.assert_array_equal(native.request_log[0]['raw_chunk'], chunk)
    assert len(native.request_log) == 1 and len(native.projection_log) == 5


def test_server_failure_is_logged_and_close_is_idempotent(observation):
    client = Client()
    policy = OpenPIPolicy(client=client)
    def fail(payload):
        raise TimeoutError('fixture receive deadline')
    client.infer = fail
    with pytest.raises(TimeoutError, match='deadline'):
        policy.act(observation)
    assert 'TimeoutError' in policy.request_log[0]['error']
    policy.close()
    policy.close()
    assert client.closes == 1
    with pytest.raises(RuntimeError, match='closed'):
        policy.act(observation)
    with pytest.raises(RuntimeError, match='closed'):
        policy.reset()


def test_official_infer_serialization_uses_bounded_handshake_and_receive(monkeypatch, observation):
    class Socket:
        def __init__(self):
            self.timeouts, self.sent, self.closes = [], [], 0
            self.replies = [msgpack_numpy.packb({'fixture': True}),
                            msgpack_numpy.packb({'actions': np.zeros((10, 7))})]
        def recv(self, timeout):
            self.timeouts.append(timeout)
            return self.replies.pop(0)
        def send(self, value):
            self.sent.append(value)
        def close(self):
            self.closes += 1
    socket = Socket()
    connections = []
    def connect(uri, **kwargs):
        connections.append((uri, kwargs))
        return socket
    monkeypatch.setattr(adapter, 'connect', connect)
    policy = OpenPIPolicy(connect_timeout=2., request_timeout=9.)
    policy.act(observation)
    assert connections[0][0] == 'ws://127.0.0.1:8000'
    assert connections[0][1]['open_timeout'] == 2.
    assert connections[0][1]['proxy'] is None
    assert socket.timeouts == [2., 9.]
    decoded = msgpack_numpy.unpackb(socket.sent[0])
    expected = adapter.prepare_openpi_observation(observation)
    for key in ('observation/image', 'observation/wrist_image', 'observation/state'):
        np.testing.assert_array_equal(decoded[key], expected[key])
    assert decoded['prompt'] == observation['language']
    policy.close()
    assert socket.closes == 1


def test_connection_failure_is_not_retried(monkeypatch):
    calls = []
    def refused(*args, **kwargs):
        calls.append(args)
        raise ConnectionRefusedError('fixture refusal')
    monkeypatch.setattr(adapter, 'connect', refused)
    with pytest.raises(ConnectionRefusedError, match='refusal'):
        OpenPIPolicy(connect_timeout=.1)
    assert len(calls) == 1


@pytest.mark.parametrize('kwargs', [{'replan_steps': 0}, {'replan_steps': 11},
                                    {'connect_timeout': 0}, {'request_timeout': float('inf')},
                                    {'native_action_bounds': 1}])
def test_invalid_configuration_rejected_before_connect(kwargs):
    with pytest.raises(ValueError):
        OpenPIPolicy(**kwargs, client=Client())
