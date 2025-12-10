using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Applies exponential moving average (EMA) smoothing to EEG values
/// with configurable time constants and normalization
/// </summary>
public class EegSmoother : MonoBehaviour
{
    [Header("Smoothing Parameters")]
    [Range(0.1f, 2f)]
    [SerializeField] private float tauSeconds = 0.5f; // Time constant for EMA
    
    [Header("Normalization")]
    [SerializeField] private bool enableNormalization = true;
    [SerializeField] private float rollingWindowSeconds = 10f;
    [SerializeField] private float outlierClampStdDev = 3f;
    
    [Header("Current Values (Smoothed)")]
    [SerializeField] private float alpha;
    [SerializeField] private float beta;
    [SerializeField] private float theta;
    [SerializeField] private float delta;
    [SerializeField] private float arousal;
    [SerializeField] private float left;
    [SerializeField] private float right;
    [SerializeField] private float front;
    [SerializeField] private float back;
    
    // Raw target values
    private float targetAlpha, targetBeta, targetTheta, targetDelta, targetArousal;
    private float targetLeft, targetRight, targetFront, targetBack;
    
    // Normalization tracking
    private class RollingStats
    {
        private Queue<(float time, float value)> samples = new Queue<(float, float)>();
        private float sum = 0f;
        private float sumSquared = 0f;
        private int count = 0;
        
        public void AddSample(float value, float time, float windowSeconds)
        {
            samples.Enqueue((time, value));
            sum += value;
            sumSquared += value * value;
            count++;
            
            // Remove old samples
            while (samples.Count > 0 && samples.Peek().time < time - windowSeconds)
            {
                var old = samples.Dequeue();
                sum -= old.value;
                sumSquared -= old.value * old.value;
                count--;
            }
        }
        
        public float GetMean()
        {
            return count > 0 ? sum / count : 0.5f;
        }
        
        public float GetStdDev()
        {
            if (count < 2) return 0.25f;
            float mean = GetMean();
            float variance = (sumSquared / count) - (mean * mean);
            return Mathf.Sqrt(Mathf.Max(0f, variance));
        }
        
        public float Normalize(float value, float clampStdDev)
        {
            if (count < 10) return Mathf.Clamp01(value); // Not enough data
            
            float mean = GetMean();
            float stdDev = GetStdDev();
            
            if (stdDev < 0.01f) return 0.5f; // No variation
            
            // Z-score normalization
            float z = (value - mean) / stdDev;
            
            // Clamp outliers
            z = Mathf.Clamp(z, -clampStdDev, clampStdDev);
            
            // Map to [0,1]
            return (z + clampStdDev) / (2f * clampStdDev);
        }
    }
    
    private Dictionary<string, RollingStats> stats = new Dictionary<string, RollingStats>();
    
    // Public properties for other components
    public float Alpha => alpha;
    public float Beta => beta;
    public float Theta => theta;
    public float Delta => delta;
    public float Arousal => arousal;
    public float Left => left;
    public float Right => right;
    public float Front => front;
    public float Back => back;
    
    void Awake()
    {
        // Initialize stats trackers
        stats["alpha"] = new RollingStats();
        stats["beta"] = new RollingStats();
        stats["theta"] = new RollingStats();
        stats["delta"] = new RollingStats();
        stats["arousal"] = new RollingStats();
        stats["left"] = new RollingStats();
        stats["right"] = new RollingStats();
        stats["front"] = new RollingStats();
        stats["back"] = new RollingStats();
        
        // Set initial values
        alpha = beta = theta = delta = 0.5f;
        arousal = left = right = front = back = 0.5f;
    }
    
    public void UpdateValues(UdpReceiver.EegPacket packet)
    {
        float time = Time.time;
        
        // Update stats and normalize if enabled
        if (enableNormalization)
        {
            stats["alpha"].AddSample(packet.alpha, time, rollingWindowSeconds);
            stats["beta"].AddSample(packet.beta, time, rollingWindowSeconds);
            stats["theta"].AddSample(packet.theta, time, rollingWindowSeconds);
            stats["delta"].AddSample(packet.delta, time, rollingWindowSeconds);
            stats["arousal"].AddSample(packet.arousal, time, rollingWindowSeconds);
            stats["left"].AddSample(packet.left, time, rollingWindowSeconds);
            stats["right"].AddSample(packet.right, time, rollingWindowSeconds);
            stats["front"].AddSample(packet.front, time, rollingWindowSeconds);
            stats["back"].AddSample(packet.back, time, rollingWindowSeconds);
            
            targetAlpha = stats["alpha"].Normalize(packet.alpha, outlierClampStdDev);
            targetBeta = stats["beta"].Normalize(packet.beta, outlierClampStdDev);
            targetTheta = stats["theta"].Normalize(packet.theta, outlierClampStdDev);
            targetDelta = stats["delta"].Normalize(packet.delta, outlierClampStdDev);
            targetArousal = stats["arousal"].Normalize(packet.arousal, outlierClampStdDev);
            targetLeft = stats["left"].Normalize(packet.left, outlierClampStdDev);
            targetRight = stats["right"].Normalize(packet.right, outlierClampStdDev);
            targetFront = stats["front"].Normalize(packet.front, outlierClampStdDev);
            targetBack = stats["back"].Normalize(packet.back, outlierClampStdDev);
        }
        else
        {
            // Direct clamping without normalization
            targetAlpha = Mathf.Clamp01(packet.alpha);
            targetBeta = Mathf.Clamp01(packet.beta);
            targetTheta = Mathf.Clamp01(packet.theta);
            targetDelta = Mathf.Clamp01(packet.delta);
            targetArousal = Mathf.Clamp01(packet.arousal);
            targetLeft = Mathf.Clamp01(packet.left);
            targetRight = Mathf.Clamp01(packet.right);
            targetFront = Mathf.Clamp01(packet.front);
            targetBack = Mathf.Clamp01(packet.back);
        }
    }
    
    void Update()
    {
        // Apply exponential moving average smoothing
        float dt = Time.deltaTime;
        float smoothingFactor = 1f - Mathf.Exp(-dt / tauSeconds);
        
        alpha = Mathf.Lerp(alpha, targetAlpha, smoothingFactor);
        beta = Mathf.Lerp(beta, targetBeta, smoothingFactor);
        theta = Mathf.Lerp(theta, targetTheta, smoothingFactor);
        delta = Mathf.Lerp(delta, targetDelta, smoothingFactor);
        arousal = Mathf.Lerp(arousal, targetArousal, smoothingFactor);
        left = Mathf.Lerp(left, targetLeft, smoothingFactor);
        right = Mathf.Lerp(right, targetRight, smoothingFactor);
        front = Mathf.Lerp(front, targetFront, smoothingFactor);
        back = Mathf.Lerp(back, targetBack, smoothingFactor);
        
        // Update global shader variables
        Shader.SetGlobalFloat("_EEG_Alpha", alpha);
        Shader.SetGlobalFloat("_EEG_Beta", beta);
        Shader.SetGlobalFloat("_EEG_Theta", theta);
        Shader.SetGlobalFloat("_EEG_Delta", delta);
        Shader.SetGlobalFloat("_EEG_Arousal", arousal);
        
        // Additional derived values
        float heat = (alpha + beta) * 0.5f;
        Shader.SetGlobalFloat("_EEG_Heat", heat);
        
        float balance = (left - right) * 0.5f + 0.5f; // [-1,1] to [0,1]
        Shader.SetGlobalFloat("_EEG_Balance", balance);
    }
    
    // Manual value injection for testing
    public void SetTestValues(float a, float b, float t, float d, float ar)
    {
        targetAlpha = Mathf.Clamp01(a);
        targetBeta = Mathf.Clamp01(b);
        targetTheta = Mathf.Clamp01(t);
        targetDelta = Mathf.Clamp01(d);
        targetArousal = Mathf.Clamp01(ar);
    }
    
    public void ResetNormalization()
    {
        foreach (var stat in stats.Values)
        {
            stat.AddSample(0.5f, Time.time, rollingWindowSeconds);
        }
    }
}