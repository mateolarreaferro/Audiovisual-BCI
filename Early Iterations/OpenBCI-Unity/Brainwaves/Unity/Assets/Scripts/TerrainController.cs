using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Maps EEG values to terrain-wide visual parameters
/// Controls skybox, fog, wind, lighting based on brain activity
/// </summary>
[RequireComponent(typeof(EegSmoother))]
public class TerrainController : MonoBehaviour
{
    [Header("Scene References")]
    [SerializeField] private WindZone windZone;
    [SerializeField] private Light directionalLight;
    [SerializeField] private Material skyboxMaterial;
    [SerializeField] private Camera mainCamera;
    
    [Header("Effect Parameters")]
    [SerializeField] private float baseFogDensity = 0.01f;
    [SerializeField] private float maxFogDensity = 0.08f;
    [SerializeField] private float baseWindStrength = 0.2f;
    [SerializeField] private float maxWindStrength = 2.5f;
    [SerializeField] private float baseLightIntensity = 0.6f;
    [SerializeField] private float maxLightIntensity = 1.4f;
    [SerializeField] private float baseSkyboxExposure = 1.0f;
    [SerializeField] private float maxSkyboxExposure = 2.5f;
    
    [Header("Color Mapping")]
    [SerializeField] private Gradient fogColorGradient;
    [SerializeField] private Gradient lightColorGradient;
    [SerializeField] private Color calmTint = new Color(0.4f, 0.6f, 1f, 1f);
    [SerializeField] private Color engagedTint = new Color(1f, 0.6f, 0.4f, 1f);
    
    [Header("IMU Camera Control")]
    [SerializeField] private bool useImuForCamera = false;
    [SerializeField] private float imuTiltScale = 15f;
    [SerializeField] private float imuSmoothSpeed = 2f;
    
    private EegSmoother eegSmoother;
    private Vector3 originalCameraRotation;
    private Vector3 targetCameraRotation;
    private float[] lastImu = new float[3];
    
    // Cache original values
    private float originalFogDensity;
    private Color originalFogColor;
    private float originalWindMain;
    private float originalWindTurbulence;
    private float originalLightIntensity;
    private Color originalLightColor;
    
    void Awake()
    {
        eegSmoother = GetComponent<EegSmoother>();
        
        // Auto-find references if not set
        if (!windZone) windZone = FindObjectOfType<WindZone>();
        if (!directionalLight) directionalLight = FindObjectOfType<Light>();
        if (!mainCamera) mainCamera = Camera.main;
        
        // Store original values
        originalFogDensity = RenderSettings.fogDensity;
        originalFogColor = RenderSettings.fogColor;
        
        if (windZone)
        {
            originalWindMain = windZone.windMain;
            originalWindTurbulence = windZone.windTurbulence;
        }
        
        if (directionalLight)
        {
            originalLightIntensity = directionalLight.intensity;
            originalLightColor = directionalLight.color;
        }
        
        if (mainCamera)
        {
            originalCameraRotation = mainCamera.transform.eulerAngles;
            targetCameraRotation = originalCameraRotation;
        }
        
        // Create default gradients if not set
        if (fogColorGradient == null || fogColorGradient.colorKeys.Length == 0)
        {
            fogColorGradient = new Gradient();
            GradientColorKey[] colorKeys = new GradientColorKey[3];
            colorKeys[0] = new GradientColorKey(new Color(0.7f, 0.8f, 1f), 0f);
            colorKeys[1] = new GradientColorKey(new Color(0.9f, 0.9f, 0.8f), 0.5f);
            colorKeys[2] = new GradientColorKey(new Color(1f, 0.7f, 0.5f), 1f);
            
            GradientAlphaKey[] alphaKeys = new GradientAlphaKey[2];
            alphaKeys[0] = new GradientAlphaKey(1f, 0f);
            alphaKeys[1] = new GradientAlphaKey(1f, 1f);
            
            fogColorGradient.SetKeys(colorKeys, alphaKeys);
        }
        
        if (lightColorGradient == null || lightColorGradient.colorKeys.Length == 0)
        {
            lightColorGradient = new Gradient();
            GradientColorKey[] colorKeys = new GradientColorKey[3];
            colorKeys[0] = new GradientColorKey(new Color(0.8f, 0.9f, 1f), 0f);
            colorKeys[1] = new GradientColorKey(Color.white, 0.5f);
            colorKeys[2] = new GradientColorKey(new Color(1f, 0.9f, 0.7f), 1f);
            
            GradientAlphaKey[] alphaKeys = new GradientAlphaKey[2];
            alphaKeys[0] = new GradientAlphaKey(1f, 0f);
            alphaKeys[1] = new GradientAlphaKey(1f, 1f);
            
            lightColorGradient.SetKeys(colorKeys, alphaKeys);
        }
        
        // Enable fog
        RenderSettings.fog = true;
        RenderSettings.fogMode = FogMode.Exponential;
    }
    
    void Update()
    {
        if (!eegSmoother) return;
        
        // Get smoothed EEG values
        float alpha = eegSmoother.Alpha;
        float beta = eegSmoother.Beta;
        float theta = eegSmoother.Theta;
        float delta = eegSmoother.Delta;
        float arousal = eegSmoother.Arousal;
        
        // Calculate derived metrics
        float focus = (beta + alpha) * 0.5f;
        float relaxation = (theta + delta) * 0.5f;
        float activity = Mathf.Clamp01(arousal * 1.2f);
        
        // Update Skybox Exposure (Alpha drives brightness)
        if (skyboxMaterial)
        {
            float skyboxExposure = Mathf.Lerp(baseSkyboxExposure, maxSkyboxExposure, alpha);
            skyboxMaterial.SetFloat("_Exposure", skyboxExposure);
            
            // Optional: tint based on arousal
            Color skyTint = Color.Lerp(calmTint, engagedTint, arousal);
            skyboxMaterial.SetColor("_Tint", skyTint);
        }
        else if (RenderSettings.skybox)
        {
            float skyboxExposure = Mathf.Lerp(baseSkyboxExposure, maxSkyboxExposure, alpha);
            RenderSettings.skybox.SetFloat("_Exposure", skyboxExposure);
        }
        
        // Update Fog (Delta drives density, arousal drives color)
        float fogDensity = Mathf.Lerp(baseFogDensity, maxFogDensity, delta);
        RenderSettings.fogDensity = fogDensity;
        
        Color fogColor = fogColorGradient.Evaluate(arousal);
        RenderSettings.fogColor = fogColor;
        
        // Update Wind (Beta drives strength)
        if (windZone)
        {
            float windMain = Mathf.Lerp(baseWindStrength, maxWindStrength, beta);
            windZone.windMain = windMain;
            
            // Turbulence based on theta
            windZone.windTurbulence = theta * 2f;
            
            // Pulse frequency based on alpha
            windZone.windPulseMagnitude = alpha * 0.5f;
            windZone.windPulseFrequency = 0.5f + beta * 2f;
            
            // Wind direction variation
            float windAngle = Time.time * (0.2f + beta * 0.5f);
            windZone.transform.rotation = Quaternion.Euler(30f + theta * 20f, windAngle * Mathf.Rad2Deg, 0f);
        }
        
        // Update Directional Light (Arousal drives intensity)
        if (directionalLight)
        {
            float lightIntensity = Mathf.Lerp(baseLightIntensity, maxLightIntensity, arousal);
            directionalLight.intensity = lightIntensity;
            
            Color lightColor = lightColorGradient.Evaluate(focus);
            directionalLight.color = lightColor;
            
            // Subtle light rotation for dynamic shadows
            float lightAngle = Time.time * 0.05f + alpha * 0.1f;
            directionalLight.transform.rotation = Quaternion.Euler(30f + delta * 15f, lightAngle * Mathf.Rad2Deg, 0f);
        }
        
        // Update ambient lighting
        RenderSettings.ambientIntensity = 0.4f + relaxation * 0.4f;
        Color ambientColor = Color.Lerp(new Color(0.3f, 0.35f, 0.4f), new Color(0.5f, 0.45f, 0.35f), focus);
        RenderSettings.ambientLight = ambientColor;
        
        // Reflection intensity based on alpha
        RenderSettings.reflectionIntensity = 0.5f + alpha * 0.5f;
        
        // Update global shader properties for custom effects
        Shader.SetGlobalFloat("_EEG_Focus", focus);
        Shader.SetGlobalFloat("_EEG_Relaxation", relaxation);
        Shader.SetGlobalFloat("_EEG_Activity", activity);
        Shader.SetGlobalVector("_EEG_Wind", new Vector4(windZone ? windZone.windMain : 0f, 
                                                        windZone ? windZone.windTurbulence : 0f, 
                                                        Time.time, 0f));
    }
    
    public void UpdateImu(float[] imu)
    {
        if (imu != null && imu.Length >= 3)
        {
            lastImu = imu;
            
            if (useImuForCamera && mainCamera)
            {
                // Map IMU to camera tilt
                targetCameraRotation = originalCameraRotation + new Vector3(
                    imu[0] * imuTiltScale,  // Pitch
                    imu[1] * imuTiltScale,  // Yaw
                    imu[2] * imuTiltScale   // Roll
                );
            }
        }
    }
    
    void LateUpdate()
    {
        // Smooth camera rotation from IMU
        if (useImuForCamera && mainCamera)
        {
            Vector3 currentRotation = mainCamera.transform.eulerAngles;
            Vector3 smoothedRotation = Vector3.Lerp(currentRotation, targetCameraRotation, 
                                                   Time.deltaTime * imuSmoothSpeed);
            mainCamera.transform.rotation = Quaternion.Euler(smoothedRotation);
        }
    }
    
    void OnDestroy()
    {
        // Restore original values
        RenderSettings.fogDensity = originalFogDensity;
        RenderSettings.fogColor = originalFogColor;
        
        if (windZone)
        {
            windZone.windMain = originalWindMain;
            windZone.windTurbulence = originalWindTurbulence;
        }
        
        if (directionalLight)
        {
            directionalLight.intensity = originalLightIntensity;
            directionalLight.color = originalLightColor;
        }
    }
    
    // Public methods for external control
    public void SetBaseFogDensity(float density)
    {
        baseFogDensity = Mathf.Clamp(density, 0f, 0.1f);
    }
    
    public void EnableImuCamera(bool enable)
    {
        useImuForCamera = enable;
        if (!enable && mainCamera)
        {
            mainCamera.transform.rotation = Quaternion.Euler(originalCameraRotation);
        }
    }
}