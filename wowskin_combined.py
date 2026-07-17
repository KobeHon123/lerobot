#!/usr/bin/env python
# Combined Visualization for WowSkin (Spatial 3D + 15-dimensional wave form)

import time
import numpy as np
import os
import collections
import sys
import argparse
from datetime import datetime

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from anyskin import AnySkinProcess

def visualize_combined(port, file=None, scaling=7.0, wave_scaling=2.0, record=False, viz_mode="3axis"):
    # Initialize the sensor stream if no file is provided
    if file is None:
        sensor_stream = AnySkinProcess(num_mags=5, port=port)
        sensor_stream.start()
        time.sleep(1.0)  # Wait for sensor stream to stabilize
        filename = "data/combined_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        load_data = np.loadtxt(file)

    pygame.init()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Read background image for spatial visualization
    bg_image_path = os.path.join(dir_path, "images/wowskin_bg.png")
    bg_image = pygame.image.load(bg_image_path)
    image_width, image_height = bg_image.get_size()

    # Fixed width for the left panel (spatial view)
    spatial_width = 400
    aspect_ratio = image_height / image_width
    spatial_height = int(spatial_width * aspect_ratio)
    
    # Scale background
    bg_image = pygame.transform.scale(bg_image, (spatial_width, spatial_height))
    
    # Chip locations relative to the 400-width image
    chip_locations = np.array([
        [201, 238],  # center
        [126, 238],  # left
        [275, 238],  # right
        [201, 163],  # up
        [201, 312],  # down
    ])
    # XY Rotations
    chip_xy_rotations = np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0])

    # Configure the window
    # Left part: 400px (spatial view) | Right part: 800px (waveform)
    waveform_width = 800
    width = spatial_width + waveform_width
    height = max(750, spatial_height) # Ensure height matches at least the required waveform height

    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("WowSkin Combined Visualization (Spatial + Waveform)")
    font = pygame.font.SysFont(None, 24)

    # Colors
    axis_colors = [(255, 80, 80), (80, 255, 80), (80, 150, 255)]
    bg_color = (25, 25, 25)
    grid_color = (60, 60, 60)
    text_color = (200, 200, 200)

    # We will keep a history of the data points based on screen width
    history_len = waveform_width
    history = collections.deque(maxlen=history_len)

    def get_baseline():
        baseline_data = sensor_stream.get_data(num_samples=5)
        baseline_data = np.array(baseline_data)[:, 1:]
        return np.mean(baseline_data, axis=0)

    if file is None:
        time.sleep(0.1)
        baseline = get_baseline()
    else:
        baseline = None

    clock = pygame.time.Clock()
    FPS = 60
    running = True
    data_len = 0
    all_recorded_data = []

    # --- Filter Setup ---
    filter_enabled = False
    filter_alpha = 0.15
    prev_filtered_data = None
    
    # --- Extended Feature Setup ---
    delta_enabled = False
    prev_sensor_data = None
    
    record_state = "IDLE"  # IDLE, CALIBRATING, RECORDING
    calibration_start_time = 0
    calibration_samples = []
    
    # --- Button Setup (Right aligned) ---
    filter_button_rect = pygame.Rect(width - 160, 10, 140, 40)
    reset_button_rect = pygame.Rect(width - 310, 10, 140, 40)
    delta_button_rect = pygame.Rect(width - 460, 10, 140, 40)
    record_button_rect = pygame.Rect(width - 610, 10, 140, 40)

    print("Combined visualizer running... Press 'Q' or close window to exit. Press 'B' to reset baseline.")

    def visualize_spatial(data_to_show):
        # Draw background with light color behind the spatial panel
        pygame.draw.rect(window, (234, 237, 232), (0, 0, spatial_width, height))
        window.blit(bg_image, (0, 0))

        data = data_to_show.reshape(-1, 3) 
        data_mag = np.linalg.norm(data, axis=1)

        # Draw each chip data
        for magid, chip_location in enumerate(chip_locations):
            if viz_mode == "magnitude":
                pygame.draw.circle(
                    window, (255, 83, 72), chip_location, data_mag[magid] / scaling
                )
            elif viz_mode == "3axis":
                # z axis
                width_circle = 2 if data[magid, -1] < 0 else 0
                pygame.draw.circle(
                    window,
                    (255, 0, 0),
                    chip_location,
                    abs(data[magid, -1]) / scaling,
                    width_circle,
                )

                # xy arrows
                arrow_start = chip_location
                rot = chip_xy_rotations[magid]
                rotation_mat = np.array([
                    [np.cos(rot), -np.sin(rot)],
                    [np.sin(rot), np.cos(rot)],
                ])
                data_xy = np.dot(rotation_mat, data[magid, :2])
                arrow_end = (
                    chip_location[0] + data_xy[0] / scaling,
                    chip_location[1] + data_xy[1] / scaling,
                )
                pygame.draw.line(window, (0, 255, 0), arrow_start, arrow_end, 2)

    while running:
        window.fill(bg_color)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if filter_button_rect.collidepoint(x, y):
                    filter_enabled = not filter_enabled
                    prev_filtered_data = None
                elif reset_button_rect.collidepoint(x, y):
                    if file is None:
                        baseline = get_baseline()
                    else:
                        baseline = np.zeros(15)
                elif delta_button_rect.collidepoint(x, y):
                    delta_enabled = not delta_enabled
                elif record_button_rect.collidepoint(x, y):
                    if record_state == "IDLE":
                        record_state = "CALIBRATING"
                        calibration_start_time = time.time()
                        calibration_samples = []
                        all_recorded_data = [] # Reset for new recording
                        print("Calibrating for 0.5s... Pls keep sensor still!")
                    elif record_state == "RECORDING":
                        record_state = "IDLE"
                        print("Recording stopped.")
                        if len(all_recorded_data) > 0:
                            os.makedirs("data", exist_ok=True)
                            fname = "data/record_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
                            np.savetxt(fname, np.array(all_recorded_data))
                            print(f"Saved recorded data to {fname}")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_b:
                    if file is None:
                        baseline = get_baseline()
                    else:
                        baseline = np.zeros(15)

        # Get sensor data
        if file is not None:
            if data_len < len(load_data):
                sensor_data = load_data[data_len]
                data_len += 1
            if baseline is None:
                baseline = np.zeros_like(sensor_data)
            raw_data = sensor_data - baseline
        else:
            sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
            
            # --- Calibration Logic ---
            if record_state == "CALIBRATING":
                calibration_samples.append(sensor_data)
                if time.time() - calibration_start_time >= 0.5:
                    baseline = np.mean(calibration_samples, axis=0) # Save as bias
                    record_state = "RECORDING"
                    print("Calibration done. Recording started: Xactual = Xraw - bias")
            
            raw_data = sensor_data - baseline

        # --- Delta Mode Logic ---
        if delta_enabled:
            if prev_sensor_data is None:
                prev_sensor_data = sensor_data
            processed_data = sensor_data - prev_sensor_data
        else:
            processed_data = raw_data
            
        prev_sensor_data = sensor_data

        if file is None:
            if record_state == "RECORDING":
                all_recorded_data.append(processed_data)
            elif record_state == "IDLE" and record: # Support argument-based auto record
                all_recorded_data.append(processed_data)

        # Apply Real-Time Exponential Moving Average (EMA) Filter
        if filter_enabled:
            if prev_filtered_data is None:
                prev_filtered_data = processed_data
            else:
                prev_filtered_data = filter_alpha * processed_data + (1 - filter_alpha) * prev_filtered_data
            data_to_show = prev_filtered_data
        else:
            data_to_show = processed_data
            prev_filtered_data = processed_data

        # -----------------------------
        # 1. DRAW SPATIAL VISUALIZATION
        # -----------------------------
        visualize_spatial(data_to_show)

        # -----------------------------
        # 2. DRAW WAVEFORM VISUALIZATION
        # -----------------------------
        history.append(data_to_show)
        hist_array = np.array(history)
        N = hist_array.shape[0]

        top_margin = 60
        bottom_margin = 50
        plot_height = height - top_margin - bottom_margin
        box_h = plot_height // 5
        
        for chip in range(5):
            y_top = top_margin + chip * box_h
            y_base = y_top + box_h // 2
            
            # Alternative background colors to explicitly show separation between sensors
            bg_rect_color = (35, 35, 35) if chip % 2 == 1 else bg_color
            pygame.draw.rect(window, bg_rect_color, (spatial_width, y_top, waveform_width, box_h))
            
            # Draw zero baseline for this chip
            pygame.draw.line(window, grid_color, (spatial_width, y_base), (width, y_base), 1)
            
            # Draw labels
            label_surf = font.render(f"Magnetometer {chip+1} (R:X, G:Y, B:Z)", True, text_color)
            window.blit(label_surf, (spatial_width + 10, y_top + 5))
            
            if N > 1:
                # Plot X, Y, Z for this chip
                for axis in range(3):
                    channel = chip * 3 + axis
                    
                    pts = []
                    for i in range(N):
                        vx = spatial_width + waveform_width - N + i
                        # Clamp the signal
                        val = hist_array[i, channel] * wave_scaling
                        max_val = (box_h // 2) - 2
                        val = np.clip(val, -max_val, max_val)
                        vy = y_base - val
                        pts.append((vx, vy))
                    
                    if len(pts) > 1:
                        pygame.draw.lines(window, axis_colors[axis], False, pts, 2)
                    
            if chip < 4:
                pygame.draw.line(window, (100, 100, 100), (spatial_width, y_top + box_h), (width, y_top + box_h), 2)

        # --- Draw Time Scale on X-Axis ---
        axis_y = height - bottom_margin
        pygame.draw.rect(window, bg_color, (spatial_width, axis_y, waveform_width, bottom_margin)) # clear bottom
        pygame.draw.line(window, (150, 150, 150), (spatial_width, axis_y), (width, axis_y), 2)
        num_ticks = 10  
        
        for i in range(num_ticks + 1):
            px = spatial_width + int(i * (waveform_width / num_ticks))
            if px >= width: 
                px = width - 2 
                
            pygame.draw.line(window, (150, 150, 150), (px, axis_y), (px, axis_y + 10), 2)
            
            time_sec = -((width - px) / FPS)
            time_label = font.render(f"{time_sec:.1f}s", True, text_color)
            label_x = max(spatial_width, min(px - time_label.get_width() // 2, width - time_label.get_width()))
            window.blit(time_label, (label_x, axis_y + 15))


        # -----------------------------
        # DRAW BUTTONS (Top Right)
        # -----------------------------
        # Filter Button
        btn_color = (0, 150, 0) if filter_enabled else (100, 100, 100)
        pygame.draw.rect(window, btn_color, filter_button_rect, border_radius=5)
        btn_text = font.render(f"Filter: {'ON' if filter_enabled else 'OFF'}", True, (255, 255, 255))
        window.blit(btn_text, (filter_button_rect.x + (filter_button_rect.width - btn_text.get_width()) // 2, 
                               filter_button_rect.y + (filter_button_rect.height - btn_text.get_height()) // 2))
        
        # Reset Button
        pygame.draw.rect(window, (180, 50, 50), reset_button_rect, border_radius=5)
        rst_text = font.render("Reset Baseline", True, (255, 255, 255))
        window.blit(rst_text, (reset_button_rect.x + (reset_button_rect.width - rst_text.get_width()) // 2, 
                               reset_button_rect.y + (reset_button_rect.height - rst_text.get_height()) // 2))

        # Delta Mode Button
        delta_color = (150, 150, 0) if delta_enabled else (100, 100, 100)
        pygame.draw.rect(window, delta_color, delta_button_rect, border_radius=5)
        d_text = font.render(f"Delta: {'ON' if delta_enabled else 'OFF'}", True, (255, 255, 255))
        window.blit(d_text, (delta_button_rect.x + (delta_button_rect.width - d_text.get_width()) // 2, 
                             delta_button_rect.y + (delta_button_rect.height - d_text.get_height()) // 2))

        # Record Button
        if record_state == "IDLE":
            rec_color = (50, 50, 180)
            rec_str = "Rec & Calib"
        elif record_state == "CALIBRATING":
            rec_color = (200, 100, 0)
            rec_str = "Calibrating..."
        else: # RECORDING
            rec_color = (200, 0, 0)
            rec_str = "STOP Rec"
            
        pygame.draw.rect(window, rec_color, record_button_rect, border_radius=5)
        rec_text = font.render(rec_str, True, (255, 255, 255))
        window.blit(rec_text, (record_button_rect.x + (record_button_rect.width - rec_text.get_width()) // 2, 
                               record_button_rect.y + (record_button_rect.height - rec_text.get_height()) // 2))

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    if file is None:
        sensor_stream.pause_streaming()
        sensor_stream.join()
        if record and len(all_recorded_data) > 0:
            os.makedirs("data", exist_ok=True)
            np.savetxt(f"{filename}.txt", np.array(all_recorded_data))
            print(f"Data saved to {filename}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WowSkin Combined 15-axis Visualization.")
    parser.add_argument("-p", "--port", type=str, default="/dev/ttyACM0", help="Serial port to microcontroller")
    parser.add_argument("-f", "--file", type=str, default=None, help="Path to load a specific data file")
    parser.add_argument("-s", "--scaling", type=float, default=7.0, help="Scaling factor for 3D arrow visualization")
    parser.add_argument("-ws", "--wave_scaling", type=float, default=2.0, help="Scaling factor for waveforms on Y-axis")
    parser.add_argument("-v", "--viz_mode", type=str, default="3axis", choices=["magnitude", "3axis"], help="Visualization mode for the spatial view")
    parser.add_argument("-r", "--record", action="store_true", help="Record data to baseline")
    args = parser.parse_args()
    
    visualize_combined(port=args.port, file=args.file, scaling=args.scaling, wave_scaling=args.wave_scaling, record=args.record, viz_mode=args.viz_mode)
