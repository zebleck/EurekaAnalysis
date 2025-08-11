float distanceToGoal = Mathf.Max(observations["DistanceToGoal"], 1e-6f); // avoid div by zero
float previousDistance = Mathf.Max(context.previousDistance, 1e-6f);
float goalThreshold = Mathf.Max(context.goalThreshold, 1e-4f);

// --- 1. Smooth distance-based reward (exponential decay teamplate)
float distanceReward = Mathf.Exp(-4f * distanceToGoal); // fast decay as get closer
rewards["distance_proximity"] = distanceReward;

// --- 2. Progress reward to encourage active movement toward the goal
float deltaProgress = previousDistance - distanceToGoal;
float progressReward = Mathf.Clamp(deltaProgress, -0.02f, 0.02f) * 5f;
rewards["progress"] = progressReward;

// --- 3. Success reward - big reward on task completion, only given once
float success = distanceToGoal < goalThreshold ? 1.0f : 0.0f;
float successReward = success > 0f ? 2.0f : 0.0f;
rewards["success_bonus"] = successReward;

// --- 4. Efficiency shaping: small time penalty to encourage speedy solutions
float timePenalty = -0.001f;
rewards["time_penalty"] = timePenalty;

// --- 5. Exploration bonus: small reward if agent has not made any progress in a while (avoid stuckness)
bool stuck = Mathf.Abs(deltaProgress) < 1e-4f;
rewards["exploration_bonus"] = stuck ? 0.002f : 0.0f;