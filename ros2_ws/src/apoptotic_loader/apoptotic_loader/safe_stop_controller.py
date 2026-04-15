import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

class SafeStopControllerNode(Node):
    def __init__(self):
        super().__init__("safe_stop_controller")
        self.declare_parameter("stop_type", "velocity_ramp")
        self.status_pub   = self.create_publisher(String,  "~/safe_stop_status", 10)
        self.alert_pub    = self.create_publisher(String,  "~/operator_alert",   10)
        self.velocity_pub = self.create_publisher(Float64, "~/velocity_command", 10)
        self.create_subscription(String, "/apoptotic_manager/lifecycle_event", self.lifecycle_cb,  10)
        self.create_subscription(String, "~/operator_clearance",               self.clearance_cb, 10)
        self.is_stopped = False
        self.get_logger().info("Safe Stop ready. Type: " + self.get_parameter("stop_type").value)

    def lifecycle_cb(self, msg):
        if ("SAFE_STOP" in msg.data or "APOPTOSIS" in msg.data) and not self.is_stopped:
            self._execute_safe_stop(msg.data)

    def clearance_cb(self, msg):
        if msg.data.upper() in ("CLEAR", "RESUME", "OK"):
            self.is_stopped = False
            self.status_pub.publish(String(data="OPERATIONS_RESUMED"))
            self.get_logger().info("Operations resumed after operator clearance.")

    def _execute_safe_stop(self, reason):
        self.is_stopped = True
        stop_type = self.get_parameter("stop_type").value
        self.get_logger().warn(f"Safe stop [{stop_type}] - {reason}")
        for v in ([0.8, 0.6, 0.4, 0.2, 0.0] if stop_type == "velocity_ramp" else [0.0]):
            self.velocity_pub.publish(Float64(data=v))
        self.status_pub.publish(String(data="SAFE_STOP_ACTIVE:" + stop_type))
        self.alert_pub.publish(String(data="OPERATOR_ALERT|reason:" + reason))
        self.get_logger().warn("Robot at rest. Awaiting operator clearance.")

def main(args=None):
    rclpy.init(args=args)
    node = SafeStopControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
