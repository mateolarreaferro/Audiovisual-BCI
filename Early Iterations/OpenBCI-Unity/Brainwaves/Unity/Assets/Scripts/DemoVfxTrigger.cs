using System.Collections;
using UnityEngine;
using UnityEngine.VFX;

/// <summary>
/// Triggers VFX effects based on EEG state changes (calm/engaged)
/// Handles cooldowns and effect management for visual feedback
/// </summary>
public class DemoVfxTrigger : MonoBehaviour
{
    [Header("VFX References")]
    [SerializeField] private VisualEffect rippleBurstVfx;
    [SerializeField] private VisualEffect dustVfx;
    [SerializeField] private VisualEffect lightningVfx;
    [SerializeField] private ParticleSystem rippleParticles; // Fallback if no VFX Graph
    [SerializeField] private ParticleSystem dustParticles;
    
    [Header("Effect Settings")]
    [SerializeField] private float calmCooldown = 2f;
    [SerializeField] private float engagedCooldown = 1f;
    [SerializeField] private float rippleDuration = 3f;
    [SerializeField] private float dustDuration = 2f;
    [SerializeField] private float lightningDuration = 0.5f;
    
    [Header("Positioning")]
    [SerializeField] private Transform effectCenter;
    [SerializeField] private float effectRadius = 50f;
    [SerializeField] private bool followCamera = true;
    
    [Header("Debug")]
    [SerializeField] private bool showDebugSpheres = false;
    
    // State tracking
    private bool lastCalmState = false;
    private bool lastEngagedState = false;
    private float lastCalmTrigger = -10f;
    private float lastEngagedTrigger = -10f;
    
    // Auto-created effects if none assigned
    private GameObject rippleEffect;
    private GameObject dustEffect;
    
    private Camera mainCamera;
    
    void Start()
    {
        mainCamera = Camera.main;
        
        // Auto-find or create effect center
        if (!effectCenter)
        {
            effectCenter = new GameObject("VFX Center").transform;
            effectCenter.SetParent(transform);
            effectCenter.localPosition = Vector3.zero;
        }
        
        // Create fallback particle effects if VFX Graph not available
        CreateFallbackEffects();
        
        // Subscribe to UDP receiver
        UdpReceiver udpReceiver = FindObjectOfType<UdpReceiver>();
        if (udpReceiver)
        {
            udpReceiver.OnPacketReceived += OnEegPacketReceived;
        }
    }
    
    void CreateFallbackEffects()
    {
        // Create ripple effect
        if (!rippleBurstVfx && !rippleParticles)
        {
            rippleEffect = new GameObject("Ripple Effect");
            rippleEffect.transform.SetParent(effectCenter);
            rippleEffect.transform.localPosition = Vector3.zero;
            
            rippleParticles = rippleEffect.AddComponent<ParticleSystem>();
            var main = rippleParticles.main;
            main.startLifetime = rippleDuration;
            main.startSpeed = 10f;
            main.startSize = 2f;
            main.startColor = new Color(0.4f, 0.8f, 1f, 0.7f);
            main.maxParticles = 100;
            
            var shape = rippleParticles.shape;
            shape.enabled = true;
            shape.shapeType = ParticleSystemShapeType.Circle;
            shape.radius = 1f;
            
            var emission = rippleParticles.emission;
            emission.enabled = true;
            emission.rateOverTime = 0f;
            var burst = new ParticleSystem.Burst(0f, 50);
            emission.SetBursts(new ParticleSystem.Burst[] { burst });
            
            var sizeOverLifetime = rippleParticles.sizeOverLifetime;
            sizeOverLifetime.enabled = true;
            AnimationCurve sizeCurve = new AnimationCurve();
            sizeCurve.AddKey(0f, 0f);
            sizeCurve.AddKey(0.2f, 1f);
            sizeCurve.AddKey(1f, 3f);
            sizeOverLifetime.size = new ParticleSystem.MinMaxCurve(1f, sizeCurve);
            
            var colorOverLifetime = rippleParticles.colorOverLifetime;
            colorOverLifetime.enabled = true;
            Gradient gradient = new Gradient();
            gradient.SetKeys(
                new GradientColorKey[] { new GradientColorKey(Color.cyan, 0f), new GradientColorKey(Color.blue, 1f) },
                new GradientAlphaKey[] { new GradientAlphaKey(0.8f, 0f), new GradientAlphaKey(0f, 1f) }
            );
            colorOverLifetime.color = gradient;
            
            rippleParticles.Stop();
        }
        
        // Create dust effect
        if (!dustVfx && !dustParticles)
        {
            dustEffect = new GameObject("Dust Effect");
            dustEffect.transform.SetParent(effectCenter);
            dustEffect.transform.localPosition = Vector3.zero;
            
            dustParticles = dustEffect.AddComponent<ParticleSystem>();
            var main = dustParticles.main;
            main.startLifetime = dustDuration;
            main.startSpeed = 5f;
            main.startSize = 1f;
            main.startColor = new Color(1f, 0.7f, 0.4f, 0.6f);
            main.maxParticles = 200;
            
            var shape = dustParticles.shape;
            shape.enabled = true;
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = new Vector3(20f, 1f, 20f);
            
            var emission = dustParticles.emission;
            emission.enabled = true;
            emission.rateOverTime = 0f;
            var burst = new ParticleSystem.Burst(0f, 100);
            emission.SetBursts(new ParticleSystem.Burst[] { burst });
            
            var velocityOverLifetime = dustParticles.velocityOverLifetime;
            velocityOverLifetime.enabled = true;
            velocityOverLifetime.space = ParticleSystemSimulationSpace.Local;
            velocityOverLifetime.y = new ParticleSystem.MinMaxCurve(3f, 8f);
            
            dustParticles.Stop();
        }
    }
    
    void OnEegPacketReceived(UdpReceiver.EegPacket packet)
    {
        bool currentCalm = packet.calm == 1;
        bool currentEngaged = packet.engaged == 1;
        
        // Detect calm state transition
        if (currentCalm && !lastCalmState && CanTriggerCalm())
        {
            TriggerCalmEffect();
        }
        
        // Detect engaged state transition
        if (currentEngaged && !lastEngagedState && CanTriggerEngaged())
        {
            TriggerEngagedEffect();
        }
        
        lastCalmState = currentCalm;
        lastEngagedState = currentEngaged;
    }
    
    bool CanTriggerCalm()
    {
        return Time.time - lastCalmTrigger >= calmCooldown;
    }
    
    bool CanTriggerEngaged()
    {
        return Time.time - lastEngagedTrigger >= engagedCooldown;
    }
    
    public void TriggerCalmEffect()
    {
        lastCalmTrigger = Time.time;
        
        UpdateEffectPosition();
        
        // Try VFX Graph first
        if (rippleBurstVfx)
        {
            rippleBurstVfx.Play();
            
            // Set VFX properties if available
            if (rippleBurstVfx.HasFloat("Radius"))
                rippleBurstVfx.SetFloat("Radius", effectRadius);
            if (rippleBurstVfx.HasFloat("Duration"))
                rippleBurstVfx.SetFloat("Duration", rippleDuration);
        }
        // Fallback to particle system
        else if (rippleParticles)
        {
            rippleParticles.Play();
        }
        
        Debug.Log("[VfxTrigger] Calm state triggered - Ripple burst");
        StartCoroutine(FlashScreenEffect(new Color(0.2f, 0.5f, 1f, 0.1f), 0.3f));
    }
    
    public void TriggerEngagedEffect()
    {
        lastEngagedTrigger = Time.time;
        
        UpdateEffectPosition();
        
        // Try VFX Graph first
        if (dustVfx)
        {
            dustVfx.Play();
        }
        // Fallback to particle system
        else if (dustParticles)
        {
            dustParticles.Play();
        }
        
        // Lightning effect
        if (lightningVfx)
        {
            lightningVfx.Play();
            StartCoroutine(StopEffectAfterDelay(lightningVfx, lightningDuration));
        }
        
        Debug.Log("[VfxTrigger] Engaged state triggered - Dust/Lightning");
        StartCoroutine(FlashScreenEffect(new Color(1f, 0.6f, 0.2f, 0.1f), 0.2f));
    }
    
    void UpdateEffectPosition()
    {
        if (followCamera && mainCamera && effectCenter)
        {
            // Position effects in front of camera
            Vector3 cameraPos = mainCamera.transform.position;
            Vector3 forward = mainCamera.transform.forward;
            effectCenter.position = cameraPos + forward * 20f + Vector3.up * -5f;
        }
    }
    
    IEnumerator StopEffectAfterDelay(VisualEffect vfx, float delay)
    {
        yield return new WaitForSeconds(delay);
        if (vfx)
            vfx.Stop();
    }
    
    IEnumerator FlashScreenEffect(Color flashColor, float duration)
    {
        // Simple screen flash by modifying fog temporarily
        Color originalFog = RenderSettings.fogColor;
        float originalIntensity = RenderSettings.ambientIntensity;
        
        RenderSettings.ambientIntensity = originalIntensity + 0.3f;
        RenderSettings.fogColor = Color.Lerp(originalFog, flashColor, 0.3f);
        
        yield return new WaitForSeconds(duration);
        
        RenderSettings.ambientIntensity = originalIntensity;
        RenderSettings.fogColor = originalFog;
    }
    
    void Update()
    {
        // Update effect positions if following camera
        if (followCamera && (rippleBurstVfx || dustVfx || lightningVfx || rippleParticles || dustParticles))
        {
            UpdateEffectPosition();
        }
    }
    
    void OnDrawGizmosSelected()
    {
        if (!showDebugSpheres) return;
        
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireSphere(effectCenter ? effectCenter.position : transform.position, effectRadius * 0.5f);
        
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(effectCenter ? effectCenter.position : transform.position, effectRadius);
    }
    
    // Manual trigger methods for testing
    [ContextMenu("Test Calm Effect")]
    public void TestCalmEffect()
    {
        TriggerCalmEffect();
    }
    
    [ContextMenu("Test Engaged Effect")]
    public void TestEngagedEffect()
    {
        TriggerEngagedEffect();
    }
    
    void OnDestroy()
    {
        // Clean up created objects
        if (rippleEffect) DestroyImmediate(rippleEffect);
        if (dustEffect) DestroyImmediate(dustEffect);
    }
}