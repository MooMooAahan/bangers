class UpgradeManager:
    def __init__(self,scorekeeper):
        self.scorekeeper = scorekeeper
        self.money = 0
        
        self.upgrades = {
    "ambulance_capacity": {"level": 0, "cost": 20, "max": 4},
    "scram_speed": {"level": 0, "cost": 30, "max": 3},
    "inspect_discount": {"level": 0, "cost": 20, "max": 2}
}

        
    def earn_money(self, amount):
        self.money += amount

    def purchase(self, name):
        upgrade = self.upgrades.get(name)
        if not upgrade:
            return False
        if upgrade["level"] >= upgrade["max"]:
            return False
        if self.money < upgrade["cost"]:
            return False

        self.money -= upgrade["cost"]
        upgrade["level"] += 1
        upgrade["cost"] = int(upgrade["cost"] * 1.5)
        self.apply_effect(name)
        
        print(f"[UPGRADE] Purchased '{name}' → Level {upgrade['level']} | Remaining money: ${self.money}")
        
        return True

    def apply_effect(self, name):
        if name == "ambulance_capacity":
            self.scorekeeper.capacity += 1
        elif name == "scram_speed":
            self.scorekeeper.scram_time_reduction = getattr(self.scorekeeper, "scram_time_reduction", 0) + 20
            print(f"[DEBUG] Scram speed upgraded! Total reduction: {self.scorekeeper.scram_time_reduction} minutes")
        elif name == "inspect_discount":
            self.scorekeeper.inspect_cost_reduction = getattr(self.scorekeeper, "inspect_cost_reduction", 0) + 5
            
            
    def get_money(self):
        return self.money

    def reset(self):
        self.money = 0
        self.upgrades = {
            "ambulance_capacity": {"level": 0, "cost": 20, "max": 5},
            "scram_speed": {"level": 0, "cost": 30, "max": 3},
            "inspect_discount": {"level": 0, "cost": 20, "max": 2}
        }
        
        # Reset the actual effects applied to scorekeeper
        self.scorekeeper.capacity = 10  # Reset to base capacity
        self.scorekeeper.scram_time_reduction = 0  # Reset scram time reduction
        self.scorekeeper.inspect_cost_reduction = 0  # Reset inspect cost reduction
        
        print("[DEBUG] UpgradeManager reset - all effects cleared.")
