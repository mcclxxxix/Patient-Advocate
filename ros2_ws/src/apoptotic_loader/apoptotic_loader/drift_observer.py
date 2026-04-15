import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
import math, random

class DriftObserverNode(Node):
    def __init__(self):
        super().__init__("drift_observer")
        self.declare_parameter("kl_threshold", 0.05)
        self.declare_parameter("sample_rate", 100)
        self.declare_parameter("early_expire_on_drift", True)
        self.kl_threshold = self.get_parameter("kl_threshold").value
        self.drift_report_pub = self.create_publisher(Float64, "~/drift_report", 10)
        self.drift_alert_pub  = self.create_publisher(String,  "~/drift_alert",  10)
        self.create_subscription(String, "/apoptotic_manager/model_status", self.status_cb, 10)
        self.baseline   = [0.1] * 10
        self.monitoring = False
        self.inf_count  = 0
        self.create_timer(5.0, self.check_drift)
        self.get_logger().info(f"Drift Observer ready. KL threshold: {self.kl_threshold}")

    def kl_divergence(self, p, q):
        eps = 1e-10
        return sum(px * math.log((px+eps)/(qx+eps)) for px,qx in zip(p,q) if px > 0)

    def status_cb(self, msg):
        if msg.data == "LOADED_AND_VERIFIED":
            self.monitoring = True
            self.inf_count  = 0
            self.get_logger().info("Drift monitoring ACTIVE")
        elif msg.data in ("DESTROYED_CLEANLY", "UNLOADED"):
            self.monitoring = False
            self.get_logger().info("Drift monitoring PAUSED")

    def check_drift(self):
        if not self.monitoring:
            return
        self.inf_count += self.get_parameter("sample_rate").value
        drift = random.uniform(0.0, 0.08)
        cur   = [max(0.001, b + random.uniform(-drift, drift)) for b in self.baseline]
        t     = sum(cur)
        cur   = [x/t for x in cur]
        kl    = self.kl_divergence(self.baseline, cur)
        self.drift_report_pub.publish(Float64(data=kl))
        tag   = "ALERT" if kl >= self.kl_threshold else "OK"
        self.get_logger().info(f"[Drift #{self.inf_count}] KL={kl:.6f} | {tag}")
        if kl >= self.kl_threshold:
            self.drift_alert_pub.publish(String(data=f"KL:{kl:.6f}:THRESHOLD:{self.kl_threshold}"))
            self.get_logger().warn(f"KL {kl:.6f} >= {self.kl_threshold} - alert sent")

def main(args=None):
    rclpy.init(args=args)
    node = DriftObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
