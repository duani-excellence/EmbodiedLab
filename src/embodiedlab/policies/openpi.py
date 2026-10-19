"""Official OpenPI websocket inference, adapted to EmbodiedLab's Policy calls."""
from collections import deque
import hashlib
from time import perf_counter
from typing import Any

import numpy as np
from openpi_client import image_tools, msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from websockets.sync.client import connect

from embodiedlab.interfaces import Action, Environment, Observation


class _TimedConnection:
    """Keep upstream infer() unchanged while bounding its blocking recv()."""

    def __init__(self, connection: Any, timeout: float) -> None:
        self.connection, self.timeout = connection, timeout

    def send(self, data: bytes) -> None:
        self.connection.send(data)

    def recv(self) -> bytes | str:
        return self.connection.recv(timeout=self.timeout)

    def close(self) -> None:
        self.connection.close()


class _BoundedClient(WebsocketClientPolicy):
    """One finite connection attempt; serialization/inference remains upstream."""

    def __init__(self, host: str, port: int, connect_timeout: float,
                 request_timeout: float) -> None:
        self.connect_timeout, self.request_timeout = connect_timeout, request_timeout
        super().__init__(host=host, port=port)

    def _wait_for_server(self) -> tuple[_TimedConnection, dict]:
        connection = connect(self._uri, compression=None, max_size=None, proxy=None,
                             open_timeout=self.connect_timeout, close_timeout=5.)
        try:
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=self.connect_timeout))
            if not isinstance(metadata, dict):
                raise ValueError('OpenPI server metadata must be a dictionary')
        except Exception:
            connection.close()
            raise
        return _TimedConnection(connection, self.request_timeout), metadata

    def close(self) -> None:
        self._ws.close()


def prepare_openpi_observation(observation: Observation) -> dict:
    """Raw 256px LIBERO RGB/state -> official pi05_libero websocket payload.

    Unlike VisualBC, this checkpoint requires a 180-degree image rotation.
    State order is eef xyz (m), eef axis-angle (rad), two finger positions (m).
    """
    images = []
    for key in ('agentview_image', 'wrist_image'):
        image = np.asarray(observation[key])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f'{key} must be raw RGB uint8 (256,256,3)')
        rotated = np.ascontiguousarray(image[::-1, ::-1])
        images.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, 224, 224)))
    state = np.asarray(observation['robot_state'], dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('robot_state must contain eight finite values in the LIBERO state order')
    prompt = observation['language']
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('language must be a nonempty native LIBERO task description')
    return {'observation/image': images[0], 'observation/wrist_image': images[1],
            'observation/state': state.copy(), 'prompt': prompt}


class OpenPIPolicy:
    """Execute five actions per official ten-action prediction, then observe again.

    Returned actions are already unnormalized LIBERO OSC deltas, not joints.
    Strict bounds are the default. Set native_action_bounds=True to project
    issued actions to [-1,1], matching the native arm input clip and preserving
    the gripper sign. This is not an additional scale or gripper threshold.
    Raw predictions and every projection are logged. reset() does not reset
    the official server's advancing model RNG.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 8000, *,
                 replan_steps: int = 5, connect_timeout: float = 10.,
                 request_timeout: float = 120., native_action_bounds: bool = False,
                 client: Any = None) -> None:
        if type(replan_steps) is not int or not 1 <= replan_steps <= 10:
            raise ValueError('replan_steps must be an integer in [1,10]')
        if not all(np.isfinite(x) and x > 0 for x in (connect_timeout, request_timeout)):
            raise ValueError('connect_timeout and request_timeout must be finite and positive')
        if type(native_action_bounds) is not bool:
            raise ValueError('native_action_bounds must be an explicit bool')
        self._client = client if client is not None else _BoundedClient(
            host, port, connect_timeout, request_timeout)
        self.server_metadata = self._client.get_server_metadata()
        self.replan_steps = replan_steps
        self.native_action_bounds = native_action_bounds
        self._closed = False
        self.reset()

    def assert_compatible(self, env: Environment) -> None:
        """Check the current raw-camera and OSC contract before real rollout."""
        metadata = env.episode_metadata()
        expected = {'image_size': 256, 'image_orientation': 'raw_robosuite_opengl',
                    'control_mode': 'osc_pose_delta_normalized', 'action_dim': 7,
                    'control_dt': .05, 'settle_steps': 10,
                    'position_delta_scale_m': .05, 'axis_angle_delta_scale_rad': .5,
                    'gripper': '-1 open, +1 close'}
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError('OpenPI environment contract mismatch: ' + ', '.join(mismatches))

    def reset(self) -> None:
        """Discard pending actions and episode logs; server RNG is not reset."""
        if self._closed:
            raise RuntimeError('OpenPI policy is closed')
        self._client.reset()  # Official websocket reset is currently a no-op.
        self._actions: deque[np.ndarray] = deque()
        self.request_log: list[dict] = []
        self.act_times_ms: list[float] = []
        self.last_raw_chunk: np.ndarray | None = None
        self.projection_log: list[dict] = []
        self.projected_action_count = 0
        self.projected_dimension_counts = [0] * 7
        self.max_action_overshoot = 0.
        self._step = 0

    def act(self, observation: Observation) -> Action:
        """Return one finite float64 (7,) action; reject unsupported model output."""
        if self._closed:
            raise RuntimeError('OpenPI policy is closed')
        started = perf_counter()
        try:
            if not self._actions:
                payload = prepare_openpi_observation(observation)
                record = {'request_step': self._step, 'prompt': payload['prompt'],
                          'state': payload['observation/state'].tolist(),
                          'image_sha256': {key: hashlib.sha256(payload[key].tobytes()).hexdigest()
                                           for key in ('observation/image', 'observation/wrist_image')}}
                self.request_log.append(record)
                rpc_started = perf_counter()
                try:
                    reply = self._client.infer(payload)
                except Exception as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                    raise
                finally:
                    record['rpc_ms'] = (perf_counter() - rpc_started) * 1000.
                self.last_raw_chunk = np.array(reply['actions'], copy=True)
                record['raw_chunk'] = self.last_raw_chunk.tolist()
                record['server_timing'] = reply.get('server_timing', {})
                record['policy_timing'] = reply.get('policy_timing', {})
                record['server_request_index'] = reply.get('server_request_index')
                record['synchronized_sample_ms'] = reply.get('synchronized_sample_ms')
                chunk = np.asarray(self.last_raw_chunk, dtype=np.float64)
                if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                    record['error'] = 'Expected a finite OpenPI action chunk of shape (10,7)'
                    raise ValueError(record['error'])
                prefix = chunk[:self.replan_steps]
                # Discarded tail actions are never executed and cannot cause a bounds rejection.
                if not self.native_action_bounds and np.any(np.abs(prefix) > 1.):
                    record['error'] = 'OpenPI action is outside [-1,1]; raw chunk retained; no clipping applied'
                    raise ValueError(record['error'])
                self._actions.extend(action.copy() for action in prefix)
            raw_action = self._actions.popleft()
            action = np.clip(raw_action, -1., 1.) if self.native_action_bounds else raw_action.copy()
            clipped_dims = np.flatnonzero(raw_action != action).tolist()
            overshoot = float(np.maximum(np.abs(raw_action) - 1., 0.).max())
            self.projection_log.append({'step': self._step, 'raw_action': raw_action.tolist(),
                                        'issued_action': action.tolist(), 'clipped_dims': clipped_dims,
                                        'maxovershoot': overshoot})
            self.projected_action_count += bool(clipped_dims)
            for dim in clipped_dims:
                self.projected_dimension_counts[dim] += 1
            self.max_action_overshoot = max(self.max_action_overshoot, overshoot)
            self._step += 1
            return action
        finally:
            self.act_times_ms.append((perf_counter() - started) * 1000.)

    def close(self) -> None:
        """Release the websocket; repeated close is harmless."""
        if not self._closed:
            try:
                self._client.close()
            finally:
                self._closed = True
                self._actions.clear()
