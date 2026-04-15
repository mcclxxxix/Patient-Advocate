import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CheckpointRegistryNode(Node):
    def __init__(self):
        super().__init__("checkpoint_registry")
        self.status_pub = self.create_publisher(String, "~/checkpoint_status",   10)
        self.verify_pub = self.create_publisher(String, "~/verification_result", 10)
        self.create_subscription(String, "~/request_checkpoint", self.handle_request, 10)
        self.registry = {
            "mock_model_v1.0":  {"sha256": "MOCK_HASH",     "path": "MOCK"},
            "welding_arm_v2.4": {"sha256": "9f86d08abc123", "path": "/opt/checkpoints/welding_arm.pt"},
        }
        self.create_timer(30.0, self.heartbeat)
        self.get_logger().info(f"Checkpoint Registry ready. Models: {list(self.registry.keys())}")
        self.heartbeat()

    def handle_request(self, msg):
        mid = msg.data.strip()
        if mid not in self.registry:
            self.verify_pub.publish(String(data="ERROR:UNKNOWN:" + mid))
            return
        self.verify_pub.publish(String(data="VERIFIED:" + mid + ":" + self.registry[mid]["sha256"]))
        self.get_logger().info("Checkpoint verified: " + mid)

    def heartbeat(self):
        self.status_pub.publish(String(data=f"REGISTRY_ALIVE:{len(self.registry)}_models"))

def main(args=None):
    rclpy.init(args=args)
    node = CheckpointRegistryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
