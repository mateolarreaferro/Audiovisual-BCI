using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Main manager that coordinates all EEG systems
/// Provides inspector controls and status display
/// </summary>
public class EegManager : MonoBehaviour
{
    [Header("Core Components")]
    [SerializeField] private UdpReceiver udpReceiver;
    [SerializeField] private EegSmoother eegSmoother;
    [SerializeField] private TerrainController terrainController;
    [SerializeField] private DemoVfxTrigger vfxTrigger;
    
    [Header("Network Settings")]
    [SerializeField] private string udpIP = "0.0.0.0";
    [SerializeField] private int udpPort = 7777;
    
    [Header("UI Elements (Optional)")]
    [SerializeField] private TextMeshProUGUI statusText;
    [SerializeField] private Slider alphaSlider;
    [SerializeField] private Slider betaSlider;
    [SerializeField] private Slider thetaSlider;
    [SerializeField] private Slider deltaSlider;
    [SerializeField] private Button calmButton;
    [SerializeField] private Button engagedButton;
    [SerializeField] private Toggle imuToggle;
    
    [Header("Test Mode")]
    [SerializeField] private bool enableTestMode = false;
    [SerializeField] private KeyCode testCalmKey = KeyCode.C;
    [SerializeField] private KeyCode testEngagedKey = KeyCode.E;
    
    private float connectionCheckInterval = 1f;
    private float lastConnectionCheck;
    
    void Awake()
    {
        // Auto-find components if not assigned
        if (!udpReceiver) udpReceiver = GetComponent<UdpReceiver>();
        if (!eegSmoother) eegSmoother = GetComponent<EegSmoother>();
        if (!terrainController) terrainController = GetComponent<TerrainController>();
        if (!vfxTrigger) vfxTrigger = GetComponent<DemoVfxTrigger>();
        
        // Find in scene if still not found
        if (!udpReceiver) udpReceiver = FindObjectOfType<UdpReceiver>();
        if (!eegSmoother) eegSmoother = FindObjectOfType<EegSmoother>();
        if (!terrainController) terrainController = FindObjectOfType<TerrainController>();
        if (!vfxTrigger) vfxTrigger = FindObjectOfType<DemoVfxTrigger>();
    }
    
    void Start()
    {
        // Subscribe to packet events
        if (udpReceiver)
        {
            udpReceiver.OnPacketReceived += OnPacketReceived;
        }
        
        // Setup UI event handlers
        SetupUI();
        
        Debug.Log($"[EegManager] Initialized - Listening on {udpIP}:{udpPort}");
    }
    
    void SetupUI()
    {
        if (calmButton)
            calmButton.onClick.AddListener(() => vfxTrigger?.TriggerCalmEffect());
            
        if (engagedButton)
            engagedButton.onClick.AddListener(() => vfxTrigger?.TriggerEngagedEffect());
            
        if (imuToggle)
            imuToggle.onValueChanged.AddListener(terrainController.EnableImuCamera);
            
        // Setup test sliders
        if (alphaSlider) alphaSlider.onValueChanged.AddListener(OnSliderChanged);
        if (betaSlider) betaSlider.onValueChanged.AddListener(OnSliderChanged);
        if (thetaSlider) thetaSlider.onValueChanged.AddListener(OnSliderChanged);
        if (deltaSlider) deltaSlider.onValueChanged.AddListener(OnSliderChanged);
    }
    
    void OnSliderChanged(float value)
    {
        if (!enableTestMode) return;
        
        // Manual control via sliders
        if (eegSmoother)
        {
            eegSmoother.SetTestValues(
                alphaSlider ? alphaSlider.value : 0.5f,
                betaSlider ? betaSlider.value : 0.5f,
                thetaSlider ? thetaSlider.value : 0.5f,
                deltaSlider ? deltaSlider.value : 0.5f,
                0.5f // arousal
            );
        }
    }
    
    void OnPacketReceived(UdpReceiver.EegPacket packet)
    {
        // Update smoothing
        if (eegSmoother)
        {
            eegSmoother.UpdateValues(packet);
        }
        
        // Update terrain effects
        if (terrainController && packet.imu != null)
        {
            terrainController.UpdateImu(packet.imu);
        }
        
        // Update UI sliders if not in test mode
        if (!enableTestMode)
        {
            UpdateUIFromPacket(packet);
        }
    }
    
    void UpdateUIFromPacket(UdpReceiver.EegPacket packet)
    {
        if (alphaSlider) alphaSlider.value = packet.alpha;
        if (betaSlider) betaSlider.value = packet.beta;
        if (thetaSlider) thetaSlider.value = packet.theta;
        if (deltaSlider) deltaSlider.value = packet.delta;
    }
    
    void Update()
    {
        // Handle test mode input
        if (enableTestMode)
        {
            if (Input.GetKeyDown(testCalmKey))
                vfxTrigger?.TriggerCalmEffect();
                
            if (Input.GetKeyDown(testEngagedKey))
                vfxTrigger?.TriggerEngagedEffect();
        }
        
        // Update status display
        if (Time.time - lastConnectionCheck >= connectionCheckInterval)
        {
            UpdateStatus();
            lastConnectionCheck = Time.time;
        }
    }
    
    void UpdateStatus()
    {
        if (!statusText) return;
        
        bool connected = udpReceiver?.IsConnected() ?? false;
        float packetRate = udpReceiver?.GetPacketRate() ?? 0f;
        float latency = udpReceiver?.GetLatencyMs() ?? 0f;
        
        string status = connected ? "CONNECTED" : "DISCONNECTED";
        string color = connected ? "#00FF00" : "#FF0000";
        
        statusText.text = $"<color={color}>{status}</color>\n" +
                         $"Rate: {packetRate:F1} Hz\n" +
                         $"Latency: {latency:F1} ms\n" +
                         $"Alpha: {(eegSmoother?.Alpha ?? 0f):F2}\n" +
                         $"Beta: {(eegSmoother?.Beta ?? 0f):F2}\n" +
                         $"Theta: {(eegSmoother?.Theta ?? 0f):F2}\n" +
                         $"Delta: {(eegSmoother?.Delta ?? 0f):F2}";
    }
    
    // Public API for external control
    public void SetTestMode(bool enabled)
    {
        enableTestMode = enabled;
        
        // Show/hide sliders based on mode
        if (alphaSlider) alphaSlider.interactable = enabled;
        if (betaSlider) betaSlider.interactable = enabled;
        if (thetaSlider) thetaSlider.interactable = enabled;
        if (deltaSlider) deltaSlider.interactable = enabled;
    }
    
    public void ResetNormalization()
    {
        eegSmoother?.ResetNormalization();
    }
    
    public void TriggerCalmPulse()
    {
        vfxTrigger?.TriggerCalmEffect();
    }
    
    public void TriggerEngagedBurst()
    {
        vfxTrigger?.TriggerEngagedEffect();
    }
    
    void OnDestroy()
    {
        if (udpReceiver)
            udpReceiver.OnPacketReceived -= OnPacketReceived;
    }
}