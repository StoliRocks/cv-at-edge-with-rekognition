"""
AWS IoT Device Shadow manager.
Handles device state synchronization with the cloud.
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import AWS IoT SDK
try:
    from awsiot import iotshadow
    from awscrt import mqtt
    AWS_IOT_SHADOW_AVAILABLE = True
except ImportError:
    AWS_IOT_SHADOW_AVAILABLE = False
    logger.warning("AWS IoT Shadow SDK not available")


@dataclass
class DeviceState:
    """Device state representation."""
    occupancy_state: str = "Unknown"
    confidence: float = 0.0
    iou: float = 0.0
    last_detection_time: float = 0.0
    state_start_time: float = 0.0
    frame_id: int = 0
    fps: float = 0.0
    device_temp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for shadow update."""
        return {
            "occupancy_state": self.occupancy_state,
            "confidence": round(self.confidence, 4),
            "iou": round(self.iou, 4),
            "last_detection_time": self.last_detection_time,
            "state_start_time": self.state_start_time,
            "frame_id": self.frame_id,
            "fps": round(self.fps, 2),
            "device_temp": round(self.device_temp, 1),
            "metadata": self.metadata,
        }


class ShadowManager:
    """
    Manages AWS IoT Device Shadow for state synchronization.
    Supports delta updates and state change callbacks.
    """

    def __init__(
        self,
        mqtt_connection,
        thing_name: str,
        on_delta_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize shadow manager.

        Args:
            mqtt_connection: Active MQTT connection
            thing_name: IoT thing name
            on_delta_callback: Callback for delta updates from cloud
        """
        if not AWS_IOT_SHADOW_AVAILABLE:
            raise RuntimeError("AWS IoT Shadow SDK not available")

        self.thing_name = thing_name
        self._connection = mqtt_connection
        self._on_delta = on_delta_callback
        self._shadow_client = None
        self._current_state: Optional[DeviceState] = None
        self._lock = threading.Lock()
        self._version = 0

    def start(self) -> bool:
        """Start shadow manager and subscribe to shadow topics."""
        logger.info(f"Starting shadow manager for thing: {self.thing_name}")

        try:
            self._shadow_client = iotshadow.IotShadowClient(self._connection)

            # Subscribe to delta updates
            delta_subscribed_future, _ = self._shadow_client.subscribe_to_shadow_delta_updated_events(
                request=iotshadow.ShadowDeltaUpdatedSubscriptionRequest(
                    thing_name=self.thing_name
                ),
                qos=mqtt.QoS.AT_LEAST_ONCE,
                callback=self._on_shadow_delta,
            )
            delta_subscribed_future.result(timeout=10.0)

            # Subscribe to get shadow responses
            get_accepted_future, _ = self._shadow_client.subscribe_to_get_shadow_accepted(
                request=iotshadow.GetShadowSubscriptionRequest(
                    thing_name=self.thing_name
                ),
                qos=mqtt.QoS.AT_LEAST_ONCE,
                callback=self._on_get_shadow_accepted,
            )
            get_accepted_future.result(timeout=10.0)

            # Subscribe to update shadow responses
            update_accepted_future, _ = self._shadow_client.subscribe_to_update_shadow_accepted(
                request=iotshadow.UpdateShadowSubscriptionRequest(
                    thing_name=self.thing_name
                ),
                qos=mqtt.QoS.AT_LEAST_ONCE,
                callback=self._on_update_shadow_accepted,
            )
            update_accepted_future.result(timeout=10.0)

            logger.info("Shadow manager started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start shadow manager: {e}")
            return False

    def _on_shadow_delta(self, delta: iotshadow.ShadowDeltaUpdatedEvent) -> None:
        """Handle shadow delta updates from cloud."""
        logger.info(f"Received shadow delta: {delta.state}")

        if self._on_delta:
            try:
                self._on_delta(delta.state)
            except Exception as e:
                logger.error(f"Error in delta callback: {e}")

    def _on_get_shadow_accepted(self, response: iotshadow.GetShadowResponse) -> None:
        """Handle get shadow response."""
        logger.debug(f"Get shadow response received, version: {response.version}")
        with self._lock:
            self._version = response.version or 0

    def _on_update_shadow_accepted(self, response: iotshadow.UpdateShadowResponse) -> None:
        """Handle update shadow response."""
        logger.debug(f"Shadow update accepted, version: {response.version}")
        with self._lock:
            self._version = response.version or 0

    def update_state(self, state: DeviceState) -> bool:
        """
        Update device shadow with new state.

        Args:
            state: New device state

        Returns:
            True if update initiated successfully
        """
        if self._shadow_client is None:
            logger.warning("Shadow client not initialized")
            return False

        try:
            state_dict = state.to_dict()

            request = iotshadow.UpdateShadowRequest(
                thing_name=self.thing_name,
                state=iotshadow.ShadowState(
                    reported=state_dict,
                ),
            )

            update_future, _ = self._shadow_client.publish_update_shadow(
                request=request,
                qos=mqtt.QoS.AT_LEAST_ONCE,
            )
            update_future.result(timeout=10.0)

            with self._lock:
                self._current_state = state

            logger.debug(f"Shadow state updated: {state.occupancy_state}")
            return True

        except Exception as e:
            logger.error(f"Failed to update shadow: {e}")
            return False

    def get_state(self) -> None:
        """Request current shadow state from cloud."""
        if self._shadow_client is None:
            return

        try:
            request = iotshadow.GetShadowRequest(thing_name=self.thing_name)
            get_future, _ = self._shadow_client.publish_get_shadow(
                request=request,
                qos=mqtt.QoS.AT_LEAST_ONCE,
            )
            get_future.result(timeout=10.0)
            logger.debug("Get shadow request sent")
        except Exception as e:
            logger.error(f"Failed to get shadow: {e}")

    @property
    def current_state(self) -> Optional[DeviceState]:
        """Get current local state."""
        with self._lock:
            return self._current_state

    @property
    def version(self) -> int:
        """Get current shadow version."""
        with self._lock:
            return self._version


class SimpleShadowManager:
    """
    Simple shadow manager using direct MQTT publishing.
    Works with any MQTT client (AWS IoT or local broker).
    """

    def __init__(
        self,
        mqtt_client,
        thing_name: str,
    ):
        """
        Initialize simple shadow manager.

        Args:
            mqtt_client: MQTT client with publish capability
            thing_name: Device/thing name
        """
        self.thing_name = thing_name
        self._client = mqtt_client
        self._current_state: Optional[DeviceState] = None
        self._previous_state: Optional[str] = None
        self._state_start_time: float = time.time()

    def update_state(self, state: DeviceState) -> bool:
        """
        Update device state via MQTT.

        Args:
            state: New device state

        Returns:
            True if published successfully
        """
        # Track state transitions
        state_changed = False
        if self._previous_state != state.occupancy_state:
            state_changed = True
            self._state_start_time = time.time()
            self._previous_state = state.occupancy_state

        state.state_start_time = self._state_start_time

        # Build shadow update message
        shadow_update = {
            "state": {
                "reported": state.to_dict()
            }
        }

        # Publish to shadow topic
        shadow_topic = f"$aws/things/{self.thing_name}/shadow/update"
        result = self._client.publish(shadow_topic, shadow_update, qos=1)

        if result:
            self._current_state = state

            # Publish state change log if state changed
            if state_changed:
                self._publish_state_change_log(state)

        return result

    def _publish_state_change_log(self, state: DeviceState) -> None:
        """Publish state change to logs topic."""
        log_topic = f"device/{self.thing_name}/logs"
        log_entry = {
            "event": "state_change",
            "timestamp": time.time(),
            "new_state": state.occupancy_state,
            "confidence": state.confidence,
            "iou": state.iou,
            "frame_id": state.frame_id,
        }
        self._client.publish(log_topic, log_entry, qos=0)

    @property
    def current_state(self) -> Optional[DeviceState]:
        """Get current state."""
        return self._current_state

    @property
    def state_duration(self) -> float:
        """Get duration of current state in seconds."""
        return time.time() - self._state_start_time
