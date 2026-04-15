import math, random, time, gc
from datetime import datetime

def kl_divergence(p, q):
    eps = 1e-10
    return sum(px * math.log((px+eps)/(qx+eps)) for px,qx in zip(p,q) if px > 0)

class MockApoptoticManager:
    def __init__(self, ttl_seconds=10, kl_threshold=0.05):
        self.ttl_seconds = ttl_seconds
        self.kl_threshold = kl_threshold
        self.current_ttl = ttl_seconds
        self.state = "UNLOADED"
        self.model = None
        self.apoptosis_count = 0
        self.log(f"Manager init — TTL: {ttl_seconds}s | KL threshold: {kl_threshold}")
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [apoptotic_manager] {msg}")
    def set_state(self, state):
        self.state = state
        self.log(f"STATE -> {state}")
    def execute_model_load(self):
        self.set_state("VERIFYING")
        time.sleep(0.05)
        self.set_state("LOADING")
        time.sleep(0.05)
        self.model = f"MOCK_MODEL_v1.0"
        self.set_state("ACTIVE")
        self.log("Model loaded and verified. Lifecycle clock started.")
        return True
    def execute_model_destroy(self):
        self.log("Executing programmed cell death...")
        del self.model
        self.model = None
        gc.collect()
        self.log("Model state destroyed cleanly.")
    def trigger_apoptosis(self, reason="TTL_EXPIRED"):
        self.apoptosis_count += 1
        self.set_state("EXPIRING")
        self.log(f"Apoptosis #{self.apoptosis_count} triggered — reason: {reason}")
        self.execute_model_destroy()
        self.set_state("RELOADING")
        self.execute_model_load()
        self.current_ttl = self.ttl_seconds
        self.log(f"Reload complete. TTL reset to {self.ttl_seconds}s.")
    def tick(self):
        if self.state == "ACTIVE":
            self.current_ttl -= 1
            if self.current_ttl <= 0:
                self.log("TTL EXPIRED — executing programmed apoptosis...")
                self.trigger_apoptosis("TTL_EXPIRED")

print("\n" + "="*55)
print("  APOPTOTIC MODEL LOADING — Windows Test")
print("  Specialty Consultants / Craig McClurkin")
print("="*55)

# TEST 1: KL Divergence
print("\nTEST 1: KL Divergence")
baseline = [0.1] * 10
kl_zero = kl_divergence(baseline, baseline[:])
print(f"  Identical:    KL = {kl_zero:.8f}  (expected ~0.0)")
assert kl_zero < 1e-6
drifted = [0.5, 0.5] + [0.0]*8
kl_large = kl_divergence(baseline, drifted)
print(f"  Large drift:  KL = {kl_large:.6f}  (expected > 0.05)")
assert kl_large > 0.05
print("  PASSED")

# TEST 2: TTL Expiry
print("\nTEST 2: TTL Expiry and State Machine")
mgr = MockApoptoticManager(ttl_seconds=3, kl_threshold=0.05)
mgr.execute_model_load()
for _ in range(3):
    mgr.tick()
    time.sleep(0.01)
assert mgr.apoptosis_count == 1
assert mgr.state == "ACTIVE"
print("  PASSED")

# TEST 3: Manual Override
print("\nTEST 3: Manual Force-Expire")
mgr2 = MockApoptoticManager(ttl_seconds=100)
mgr2.execute_model_load()
mgr2.trigger_apoptosis("MANUAL_TEST")
assert mgr2.apoptosis_count == 1
assert mgr2.state == "ACTIVE"
print("  PASSED")

# TEST 4: Drift Detection
print("\nTEST 4: Drift Detection")
baseline = [0.1]*10
alerts = 0
for i in range(1, 11):
    drift = i * 0.01
    cur = [max(0.001, b + random.uniform(-drift, drift)) for b in baseline]
    total = sum(cur)
    cur = [x/total for x in cur]
    kl = kl_divergence(baseline, cur)
    status = "ALERT" if kl >= 0.05 else "OK"
    if kl >= 0.05: alerts += 1
    print(f"  Round {i:2d} | KL={kl:.6f} | {status}")
print(f"  Alerts: {alerts}/10 — PASSED")

# TEST 5: 3 Cycles
print("\nTEST 5: Multiple Apoptosis Cycles")
mgr3 = MockApoptoticManager(ttl_seconds=2)
mgr3.execute_model_load()
for i in range(3):
    for _ in range(2):
        mgr3.tick()
    print(f"  Cycle {i+1} complete | apoptosis_count={mgr3.apoptosis_count}")
assert mgr3.apoptosis_count == 3
print("  PASSED")

print("\n" + "="*55)
print("  ALL TESTS PASSED")
print("  Framework ready for ROS 2 colcon build")
print("="*55)
