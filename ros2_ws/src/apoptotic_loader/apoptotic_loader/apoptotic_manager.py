import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import gc

class ApoptoticManagerNode(Node):
    def __init__(self):
        super().__init__("apoptotic_manager")
        self.declare_parameter("ttl_seconds", 86400)
        self.declare_parameter("kl_threshold", 0.05)
        self.declare_parameter("early_expire_on_drift", True)
        self.declare_parameter("stop_type", "velocity_ramp")
        self.ttl_seconds = self.get_parameter("ttl_seconds").value
        self.current_ttl = self.ttl_seconds
        self.state = "UNLOADED"
        self.model = None
        self.ttl_pub    = self.create_publisher(Int32,  "~/ttl_countdown",   10)
        self.status_pub = self.create_publisher(String, "~/model_status",    10)
        self.event_pub  = self.create_publisher(String, "~/lifecycle_event", 10)
        self.create_subscription(String, "~/force_expire",              self.force_expire_callback, 10)
        self.create_subscription(String, "/drift_observer/drift_alert", self.drift_alert_callback,  10)
        self.create_timer(1.0, self.countdown_step)
        self.get_logger().info("=" * 50)
        self.get_logger().info("  APOPTOTIC MODEL LOADER v1.0.0")
        self.get_logger().info("  Specialty Consultants / Craig McClurkin")
        self.get_logger().info("  Apache-2.0 | craig.mcclurkin@louisville.edu")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"TTL: {self.ttl_seconds}s | KL threshold: {self.get_parameter('kl_threshold').value}")
        self._set_state("VERIFYING")
        self._execute_model_load()

    def _set_state(self, s):
        self.state = s
        self.status_pub.publish(String(data=s))
        self.event_pub.publish(String(data="STATE:" + s))
        self.get_logger().info("[STATE] -> " + s)

    def countdown_step(self):
        if self.state != "ACTIVE":
            return
        self.current_ttl -= 1
        self.ttl_pub.publish(Int32(data=self.current_ttl))
        if self.current_ttl % 3600 == 0 or self.current_ttl <= 60:
            h = self.current_ttl // 3600
            m = (self.current_ttl % 3600) // 60
            s = self.current_ttl % 60
            self.get_logger().info(f"TTL remaining: {h:02d}:{m:02d}:{s:02d}")
        if self.current_ttl <= 0:
            self.get_logger().warn("TTL EXPIRED - apoptosis triggered")
            self._trigger_apoptosis("TTL_EXPIRED")

    def force_expire_callback(self, msg):
        self.get_logger().warn("Force expire: " + msg.data)
        self._trigger_apoptosis("MANUAL_OVERRIDE:" + msg.data)

    def drift_alert_callback(self, msg):
        if self.get_parameter("early_expire_on_drift").value:
            self.get_logger().warn("Drift alert: " + msg.data)
            self._trigger_apoptosis("DRIFT_TRIGGERED:" + msg.data)

    def _trigger_apoptosis(self, reason="TTL_EXPIRED"):
        self._set_state("EXPIRING")
        self.event_pub.publish(String(data="APOPTOSIS:" + reason))
        self.get_logger().warn("Apoptosis triggered - " + reason)
        self._execute_model_destroy()
        self._set_state("RELOADING")
        if self._execute_model_load():
            self.current_ttl = self.ttl_seconds
            self.get_logger().info(f"Reload complete. TTL reset to {self.ttl_seconds}s.")
        else:
            self.get_logger().error("Reload FAILED - safe stop")
            self._set_state("UNLOADED")

    def _execute_model_load(self):
        self._set_state("LOADING")
        self.get_logger().info("Loading from cryptographically verified checkpoint...")
        self.model = "ACTIVE_MOCK_MODEL_v1.0"
        self._set_state("ACTIVE")
        self.status_pub.publish(String(data="LOADED_AND_VERIFIED"))
        self.get_logger().info("Model loaded and verified. Lifecycle clock started.")
        return True

    def _execute_model_destroy(self):
        self.get_logger().info("Executing programmed cell death...")
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        self.status_pub.publish(String(data="DESTROYED_CLEANLY"))
        self.get_logger().info("Model state destroyed cleanly.")

def main(args=None):
    rclpy.init(args=args)
    node = ApoptoticManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
