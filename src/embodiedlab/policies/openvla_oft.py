"""OFT-specific eight-action execution; preprocessing stays in its model process."""
from collections import deque
import hashlib
from time import perf_counter
from typing import Any

import numpy as np

from embodiedlab.interfaces import Observation, Environment
from embodiedlab.policies.openpi import _BoundedClient


class OpenVLAOFTPolicy:
    """Send raw RGB/state over the shared transport; execute all eight OFT actions.

    The server uses OFT's own TF/crop/state/prompt and gripper transforms.
    Reusing msgpack transport does not reuse OpenPI's input/action protocol.
    """

    def __init__(self, *, suite: str, host: str = '127.0.0.1', port: int = 8007,
                 connect_timeout: float = 10., request_timeout: float = 120.,
                 client: Any = None) -> None:
        if suite not in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
            raise ValueError('Joint OFT checkpoint supports the four trained LIBERO suites')
        if not all(np.isfinite(x) and x > 0 for x in (connect_timeout, request_timeout)):
            raise ValueError('Timeouts must be finite and positive')
        self.suite, self.replan_steps = suite, 8
        self._client = client if client is not None else _BoundedClient(
            host, port, connect_timeout, request_timeout)
        self.server_metadata = self._client.get_server_metadata()
        self._closed = False
        self.reset()

    def assert_compatible(self, env: Environment) -> None:
        """Require raw cameras and native OSC; suite selects official statistics."""
        expected = {'suite': self.suite, 'image_size': 256, 'action_dim': 7,
                    'image_orientation': 'raw_robosuite_opengl', 'settle_steps': 10,
                    'control_mode': 'osc_pose_delta_normalized', 'control_dt': .05,
                    'position_delta_scale_m': .05, 'axis_angle_delta_scale_rad': .5,
                    'gripper': '-1 open, +1 close'}
        metadata = env.episode_metadata()
        differences = [key for key, value in expected.items() if metadata.get(key) != value]
        if differences:
            raise ValueError('OFT environment mismatch: ' + ', '.join(differences))

    def reset(self) -> None:
        """Clear pending actions/logs without resetting the loaded server model."""
        if self._closed:
            raise RuntimeError('OFT policy is closed')
        self._client.reset()
        self._actions: deque[np.ndarray] = deque()
        self.request_log: list[dict] = []
        self.act_times_ms: list[float] = []
        self.projection_log: list[dict] = []
        self.projected_action_count, self.max_action_overshoot = 0, 0.
        self.projected_dimension_counts = [0] * 7
        self._step = 0

    def act(self, observation: Observation) -> np.ndarray:
        """Return final float64 (7,) LIBERO action; never convert gripper twice."""
        if self._closed:
            raise RuntimeError('OFT policy is closed')
        started = perf_counter()
        try:
            if not self._actions:
                images = {}
                for key in ('agentview_image', 'wrist_image'):
                    image = np.asarray(observation[key])
                    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                        raise ValueError(f'{key} requires raw uint8 RGB (256,256,3)')
                    images[key] = np.ascontiguousarray(image)
                state = np.asarray(observation['robot_state'], dtype=np.float64)
                prompt = observation['language']
                if state.shape != (8,) or not np.isfinite(state).all():
                    raise ValueError('OFT requires eight finite native state values')
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError('OFT requires native task language')
                payload = {**images, 'robot_state': state.copy(), 'prompt': prompt, 'suite': self.suite}
                record = {'request_step': self._step, 'suite': self.suite, 'prompt': prompt,
                          'state': state.tolist(), 'raw_image_sha256': {
                              key: hashlib.sha256(image.tobytes()).hexdigest() for key, image in images.items()}}
                self.request_log.append(record)
                rpc_start = perf_counter()
                try:
                    reply = self._client.infer(payload)
                except Exception as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                    raise
                finally:
                    record['rpc_ms'] = (perf_counter() - rpc_start) * 1000.
                chunk = np.asarray(reply['actions'], dtype=np.float64)
                raw = np.asarray(reply['raw_model_chunk'], dtype=np.float64)
                record.update(raw_chunk=raw.tolist(), issued_chunk=chunk.tolist(),
                              server_request_index=reply.get('server_request_index'),
                              synchronized_sample_ms=reply.get('synchronized_sample_ms'),
                              policy_timing=reply.get('policy_timing', {}),
                              server_preprocessing=reply.get('preprocessing', {}))
                if (chunk.shape != (8, 7) or raw.shape != (8, 7)
                        or not np.isfinite(chunk).all() or not np.isfinite(raw).all()):
                    raise ValueError('OFT requires finite raw/final chunks of shape (8,7)')
                if np.any(np.abs(chunk) > 1.):
                    record['error'] = 'Official OFT action outside [-1,1]; no clipping applied'
                    raise ValueError(record['error'])
                self._actions.extend(action.copy() for action in chunk)
            self._step += 1
            return self._actions.popleft()
        finally:
            self.act_times_ms.append((perf_counter() - started) * 1000.)

    def close(self) -> None:
        """Release this connection; repeated close is harmless."""
        if not self._closed:
            try:
                self._client.close()
            finally:
                self._closed = True
                self._actions.clear()
