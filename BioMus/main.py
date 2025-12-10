# main.py
from typing import Optional
import glob
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import asyncio

from openbci_service import GanglionService

app = FastAPI(title="Ganglion Studio")

# Templates
templates = Jinja2Templates(directory="templates")

# Global service instance
service = GanglionService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------- REST API for control --------

class ConnectParams:
    serial_port: Optional[str]
    mac_address: Optional[str]
    timeout: int


@app.post("/api/connect")
async def api_connect(payload: dict):
    serial_port = payload.get("serial_port", "")
    mac_address = payload.get("mac_address", "")
    timeout = int(payload.get("timeout", 15))
    try:
        service.connect(serial_port=serial_port, mac_address=mac_address, timeout=timeout)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/disconnect")
async def api_disconnect():
    service.disconnect()
    return {"status": "ok"}


@app.get("/api/status")
async def api_status():
    return {
        "connected": service.connected,
        "streaming": service.streaming,
    }


@app.get("/api/ports")
async def api_list_ports():
    """
    List available serial ports and Bluetooth devices for OpenBCI.
    Returns both serial ports and Bluetooth MAC addresses.
    """
    ports = []
    bluetooth_devices = []

    if sys.platform == "darwin":  # macOS
        # Look for USB serial devices (BLED112 dongle)
        ports.extend(glob.glob("/dev/tty.usbmodem*"))
        ports.extend(glob.glob("/dev/tty.usbserial*"))
        ports.extend(glob.glob("/dev/cu.usbmodem*"))
        ports.extend(glob.glob("/dev/cu.usbserial*"))

        # Check for Bluetooth Ganglion devices
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPBluetoothDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.split('\n')

            # Look for Ganglion devices
            for i, line in enumerate(lines):
                if 'ganglion' in line.lower():
                    # Look for MAC address in nearby lines
                    for j in range(max(0, i-5), min(len(lines), i+10)):
                        if 'Address:' in lines[j]:
                            mac = lines[j].split('Address:')[1].strip()
                            bluetooth_devices.append({
                                "name": line.strip().rstrip(':'),
                                "mac": mac,
                                "type": "bluetooth"
                            })
                            break
        except Exception as e:
            print(f"Bluetooth scan error: {e}")

    elif sys.platform.startswith("linux"):  # Linux
        ports.extend(glob.glob("/dev/ttyUSB*"))
        ports.extend(glob.glob("/dev/ttyACM*"))
    elif sys.platform == "win32":  # Windows
        import serial.tools.list_ports
        detected = serial.tools.list_ports.comports()
        ports = [port.device for port in detected]

    # Remove duplicates and sort
    ports = sorted(list(set(ports)))

    return {
        "ports": ports,
        "bluetooth": bluetooth_devices,
        "count": len(ports) + len(bluetooth_devices),
        "hint": "For Bluetooth: Pair your Ganglion in System Settings → Bluetooth first" if len(bluetooth_devices) == 0 and sys.platform == "darwin" else None
    }


@app.post("/api/start")
async def api_start(payload: dict):
    try:
        buffer_size = int(payload.get("buffer_size", 45000))
        service.start_stream(buffer_size=buffer_size)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/stop")
async def api_stop():
    service.stop_stream()
    return {"status": "ok"}


@app.post("/api/test_signal")
async def api_test_signal(payload: dict):
    on = bool(payload.get("on", True))
    if on:
        service.send_test_signal_on()
    else:
        service.send_test_signal_off()
    return {"status": "ok"}


@app.post("/api/osc_config")
async def api_osc_config(payload: dict):
    ip = payload.get("ip", "127.0.0.1")
    port = int(payload.get("port", 9000))
    enabled = bool(payload.get("enabled", False))
    send_raw = bool(payload.get("send_raw", True))
    send_bands = bool(payload.get("send_bands", False))
    service.configure_osc(ip, port, enabled, send_raw, send_bands)
    return {"status": "ok"}


# -------- WebSocket for stream --------

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        # Expect initial config message
        init_msg = await ws.receive_json()
        mode = init_msg.get("mode", "timeseries")
        window_sec = float(init_msg.get("window_sec", 4.0))
        send_interval_ms = int(init_msg.get("interval_ms", 100))

        while True:
            if not (service.connected and service.streaming):
                await asyncio.sleep(0.5)
                continue

            if mode == "timeseries":
                channels, data = service.get_timeseries_window(window_sec=window_sec)
                if channels:
                    # push OSC
                    service.osc_push_timeseries(channels, data)
                    await ws.send_json(
                        {
                            "type": "timeseries",
                            "channels": channels,
                            "data": data,
                        }
                    )

            elif mode == "bands":
                channels, band_names, values = service.get_band_powers(window_sec=window_sec)
                if channels:
                    service.osc_push_bands(channels, band_names, values)
                    await ws.send_json(
                        {
                            "type": "bands",
                            "channels": channels,
                            "bands": band_names,
                            "values": values,
                        }
                    )

            await asyncio.sleep(send_interval_ms / 1000.0)

    except WebSocketDisconnect:
        return
    except Exception as e:
        # send error then close
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        finally:
            await ws.close()
