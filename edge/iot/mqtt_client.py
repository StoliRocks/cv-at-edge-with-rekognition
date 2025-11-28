"""
AWS IoT MQTT client for edge device communication.
Uses AWS IoT Device SDK v2 for secure, reliable messaging.
"""

import json
import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import AWS IoT SDK
try:
    from awscrt import io, mqtt
    from awsiot import mqtt_connection_builder
    AWS_IOT_SDK_AVAILABLE = True
except ImportError:
    AWS_IOT_SDK_AVAILABLE = False
    logger.warning(
        "AWS IoT SDK not available. Install with: "
        "pip install awsiotsdk"
    )


@dataclass
class MQTTConfig:
    """MQTT connection configuration."""
    endpoint: str
    cert_path: str
    key_path: str
    root_ca_path: str
    client_id: str
    region: str = "us-east-1"
    keep_alive_secs: int = 30
    ping_timeout_ms: int = 3000
    protocol_operation_timeout_ms: int = 5000


class IoTClient:
    """
    AWS IoT Core MQTT client with automatic reconnection.
    Provides reliable message publishing and subscription.
    """

    def __init__(self, config: MQTTConfig):
        """
        Initialize IoT client.

        Args:
            config: MQTT connection configuration
        """
        if not AWS_IOT_SDK_AVAILABLE:
            raise RuntimeError(
                "AWS IoT SDK not available. Install with:\n"
                "pip install awsiotsdk"
            )

        self.config = config
        self._connection = None
        self._connected = False
        self._lock = threading.Lock()
        self._message_queue: Queue = Queue(maxsize=100)
        self._callbacks: Dict[str, Callable] = {}

        # Validate certificate paths
        self._validate_certs()

    def _validate_certs(self) -> None:
        """Validate that certificate files exist."""
        cert_path = Path(self.config.cert_path)
        key_path = Path(self.config.key_path)
        root_ca_path = Path(self.config.root_ca_path)

        missing = []
        if not cert_path.exists():
            missing.append(f"Certificate: {cert_path}")
        if not key_path.exists():
            missing.append(f"Private key: {key_path}")
        if not root_ca_path.exists():
            missing.append(f"Root CA: {root_ca_path}")

        if missing:
            raise FileNotFoundError(
                f"Missing certificate files:\n" + "\n".join(missing)
            )

    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to AWS IoT Core.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        logger.info(f"Connecting to AWS IoT: {self.config.endpoint}")

        try:
            # Create MQTT connection
            self._connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.config.endpoint,
                cert_filepath=self.config.cert_path,
                pri_key_filepath=self.config.key_path,
                ca_filepath=self.config.root_ca_path,
                client_id=self.config.client_id,
                clean_session=False,
                keep_alive_secs=self.config.keep_alive_secs,
                ping_timeout_ms=self.config.ping_timeout_ms,
                protocol_operation_timeout_ms=self.config.protocol_operation_timeout_ms,
                on_connection_interrupted=self._on_connection_interrupted,
                on_connection_resumed=self._on_connection_resumed,
            )

            # Connect
            connect_future = self._connection.connect()
            connect_future.result(timeout=timeout)

            self._connected = True
            logger.info("Connected to AWS IoT Core")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to AWS IoT: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from AWS IoT Core."""
        if self._connection is not None:
            try:
                disconnect_future = self._connection.disconnect()
                disconnect_future.result(timeout=10.0)
                logger.info("Disconnected from AWS IoT Core")
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
            finally:
                self._connected = False

    def _on_connection_interrupted(self, connection, error, **kwargs):
        """Handle connection interruption."""
        logger.warning(f"Connection interrupted: {error}")
        self._connected = False

    def _on_connection_resumed(self, connection, return_code, session_present, **kwargs):
        """Handle connection resumed."""
        logger.info(f"Connection resumed (session_present: {session_present})")
        self._connected = True

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """
        Publish message to MQTT topic.

        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON serialized)
            qos: Quality of Service (0, 1, or 2)
            retain: Retain message on broker

        Returns:
            True if published successfully
        """
        if not self._connected or self._connection is None:
            logger.warning("Not connected, cannot publish")
            return False

        try:
            message = json.dumps(payload)
            qos_enum = mqtt.QoS.AT_LEAST_ONCE if qos >= 1 else mqtt.QoS.AT_MOST_ONCE

            publish_future, _ = self._connection.publish(
                topic=topic,
                payload=message,
                qos=qos_enum,
            )
            publish_future.result(timeout=10.0)

            logger.debug(f"Published to {topic}: {len(message)} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return False

    def subscribe(
        self,
        topic: str,
        callback: Callable[[str, Dict[str, Any]], None],
        qos: int = 1,
    ) -> bool:
        """
        Subscribe to MQTT topic.

        Args:
            topic: MQTT topic (supports wildcards + and #)
            callback: Function to call when message received (topic, payload)
            qos: Quality of Service

        Returns:
            True if subscribed successfully
        """
        if not self._connected or self._connection is None:
            logger.warning("Not connected, cannot subscribe")
            return False

        def message_handler(topic, payload, **kwargs):
            try:
                data = json.loads(payload)
                callback(topic, data)
            except json.JSONDecodeError:
                callback(topic, {"raw": payload.decode()})
            except Exception as e:
                logger.error(f"Error in subscription callback: {e}")

        try:
            qos_enum = mqtt.QoS.AT_LEAST_ONCE if qos >= 1 else mqtt.QoS.AT_MOST_ONCE

            subscribe_future, _ = self._connection.subscribe(
                topic=topic,
                qos=qos_enum,
                callback=message_handler,
            )
            subscribe_future.result(timeout=10.0)

            self._callbacks[topic] = callback
            logger.info(f"Subscribed to {topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
            return False

    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from MQTT topic.

        Args:
            topic: MQTT topic

        Returns:
            True if unsubscribed successfully
        """
        if not self._connected or self._connection is None:
            return False

        try:
            unsubscribe_future, _ = self._connection.unsubscribe(topic=topic)
            unsubscribe_future.result(timeout=10.0)

            self._callbacks.pop(topic, None)
            logger.info(f"Unsubscribed from {topic}")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe from {topic}: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to AWS IoT."""
        return self._connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class LocalMQTTClient:
    """
    Local MQTT client for testing without AWS IoT.
    Uses paho-mqtt for local broker connections.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        client_id: str = "jetson-edge",
    ):
        """
        Initialize local MQTT client.

        Args:
            broker: MQTT broker hostname
            port: MQTT broker port
            client_id: Client identifier
        """
        try:
            import paho.mqtt.client as mqtt_client
            self._mqtt = mqtt_client
        except ImportError:
            raise RuntimeError(
                "paho-mqtt not available. Install with:\n"
                "pip install paho-mqtt"
            )

        self.broker = broker
        self.port = port
        self.client_id = client_id
        self._client = None
        self._connected = False
        self._callbacks: Dict[str, Callable] = {}

    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to local MQTT broker."""
        logger.info(f"Connecting to local broker: {self.broker}:{self.port}")

        try:
            self._client = self._mqtt.Client(client_id=self.client_id)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()

            # Wait for connection
            start = time.time()
            while not self._connected and (time.time() - start) < timeout:
                time.sleep(0.1)

            return self._connected

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from broker."""
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
            logger.info("Disconnected from local broker")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("Connected to local MQTT broker")
        else:
            logger.error(f"Connection failed with code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.warning(f"Disconnected from broker (rc: {rc})")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            payload = {"raw": msg.payload.decode()}

        for pattern, callback in self._callbacks.items():
            if self._topic_matches(pattern, topic):
                try:
                    callback(topic, payload)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches pattern (supports + and # wildcards)."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for i, p in enumerate(pattern_parts):
            if p == "#":
                return True
            if i >= len(topic_parts):
                return False
            if p != "+" and p != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """Publish message to topic."""
        if not self._connected or self._client is None:
            return False

        try:
            message = json.dumps(payload)
            result = self._client.publish(topic, message, qos=qos, retain=retain)
            return result.rc == 0
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return False

    def subscribe(
        self,
        topic: str,
        callback: Callable[[str, Dict[str, Any]], None],
        qos: int = 1,
    ) -> bool:
        """Subscribe to topic."""
        if not self._connected or self._client is None:
            return False

        try:
            result, _ = self._client.subscribe(topic, qos=qos)
            if result == 0:
                self._callbacks[topic] = callback
                logger.info(f"Subscribed to {topic}")
                return True
            return False
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
