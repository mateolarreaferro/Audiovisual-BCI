using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class AccelerometerCamera : MonoBehaviour
{
    [Header("Camera Settings")]
    public float tiltSensitivity = 30f;
    public float smoothing = 2f;
    
    [Header("Network")]
    public int accelerometerPort = 12346;
    
    private Vector3 targetRotation;
    private Vector3 currentAccel;
    
    private UdpClient accelUdpClient;
    private Thread accelUdpThread;
    private bool isRunning = false;
    
    void Start()
    {
        targetRotation = transform.eulerAngles;
        StartAccelerometerListener();
    }
    
    void StartAccelerometerListener()
    {
        try
        {
            accelUdpClient = new UdpClient(accelerometerPort);
            isRunning = true;
            accelUdpThread = new Thread(new ThreadStart(AccelerometerUDPListener));
            accelUdpThread.Start();
            Debug.Log($"Accelerometer UDP Listener started on port {accelerometerPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to start accelerometer UDP listener: {e.Message}");
        }
    }
    
    void AccelerometerUDPListener()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
        
        while (isRunning)
        {
            try
            {
                byte[] data = accelUdpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);
                
                // Parse format: "x,y,z" e.g., "0.1,-0.2,0.9"
                string[] parts = message.Split(',');
                if (parts.Length == 3)
                {
                    if (float.TryParse(parts[0], out float x) &&
                        float.TryParse(parts[1], out float y) &&
                        float.TryParse(parts[2], out float z))
                    {
                        currentAccel = new Vector3(x, y, z);
                    }
                }
            }
            catch (System.Exception e)
            {
                if (isRunning)
                    Debug.LogError($"Accelerometer UDP receive error: {e.Message}");
            }
        }
    }
    
    void Update()
    {
        UpdateCameraTilt();
    }
    
    void UpdateCameraTilt()
    {
        // Convert accelerometer data to camera rotation
        // X axis tilt (pitch) - phone forward/back tilt affects camera X rotation
        // Y axis tilt (roll) - phone left/right tilt affects camera Z rotation
        
        float pitch = -currentAccel.y * tiltSensitivity;  // Forward/back
        float roll = currentAccel.x * tiltSensitivity;    // Left/right
        
        targetRotation = new Vector3(
            Mathf.Clamp(30f + pitch, -60f, 90f),  // Base 30° pitch + accel adjustment
            0f,  // No yaw change
            Mathf.Clamp(roll, -45f, 45f)  // Roll limited to ±45°
        );
        
        // Smooth interpolation
        transform.rotation = Quaternion.Slerp(
            transform.rotation, 
            Quaternion.Euler(targetRotation), 
            Time.deltaTime * smoothing
        );
    }
    
    void OnDestroy()
    {
        isRunning = false;
        if (accelUdpThread != null && accelUdpThread.IsAlive)
        {
            accelUdpThread.Join(1000);
        }
        if (accelUdpClient != null)
        {
            accelUdpClient.Close();
        }
    }
    
    void OnGUI()
    {
        GUI.Label(new Rect(10, 90, 300, 20), $"Accel: {currentAccel.x:F2}, {currentAccel.y:F2}, {currentAccel.z:F2}");
        GUI.Label(new Rect(10, 110, 300, 20), $"Camera: {targetRotation.x:F1}°, {targetRotation.y:F1}°, {targetRotation.z:F1}°");
    }
}