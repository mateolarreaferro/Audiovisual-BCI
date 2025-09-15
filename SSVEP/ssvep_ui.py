#!/usr/bin/env python3
"""SSVEP User Interface Module - Handles all visual display components"""

import pygame
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class SSVEPVisualInterface:
    """Manages all visual interface components for SSVEP system"""
    
    def __init__(self, frequencies, labels, fullscreen=False):
        """
        Initialize the visual interface
        
        Args:
            frequencies: List of target frequencies
            labels: List of labels for each target
            fullscreen: Whether to run in fullscreen mode
        """
        # Visual parameters
        self.frequencies = frequencies
        self.labels = labels
        self.fullscreen = fullscreen
        self.window_size = (1024, 600)
        
        # Visual elements
        self.box_size = 250
        self.separation = 500
        
        # Colors
        self.bg_color = (128, 128, 128)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 100, 255)
        self.yellow = (255, 255, 0)
        
        # Display elements
        self.screen = None
        self.font = None
        self.small_font = None
        self.large_font = None
        self.clock = None
        
        # Box positions
        self.left_pos = None
        self.right_pos = None
        
        # Initialize Pygame
        pygame.init()
    
    def setup_display(self):
        """Initialize Pygame display and calculate positions"""
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.window_size = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode(self.window_size)
        
        pygame.display.set_caption("SSVEP BCI - Integrated System")
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Calculate positions
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2
        
        self.left_pos = (center_x - self.separation // 2 - self.box_size // 2,
                        center_y - self.box_size // 2)
        self.right_pos = (center_x + self.separation // 2 - self.box_size // 2,
                         center_y - self.box_size // 2)
    
    def clear_screen(self):
        """Clear the screen with background color"""
        self.screen.fill(self.bg_color)
    
    def draw_title(self, title, subtitle=None):
        """Draw title at top of screen"""
        text = self.large_font.render(title, True, self.white)
        text_rect = text.get_rect(centerx=self.window_size[0]//2, y=20)
        self.screen.blit(text, text_rect)
        
        if subtitle:
            text = self.font.render(subtitle, True, self.yellow)
            text_rect = text.get_rect(centerx=self.window_size[0]//2, y=70)
            self.screen.blit(text, text_rect)
    
    def draw_instructions(self, instructions):
        """Draw instruction lines"""
        y_offset = 100
        for line in instructions:
            text = self.small_font.render(line, True, self.white)
            text_rect = text.get_rect(centerx=self.window_size[0]//2, y=y_offset)
            self.screen.blit(text, text_rect)
            y_offset += 25
    
    def draw_progress_bar(self, progress, x=None, y=200, width=400, height=20):
        """Draw a progress bar"""
        if x is None:
            x = (self.window_size[0] - width) // 2
        
        # Background
        pygame.draw.rect(self.screen, self.black, (x, y, width, height))
        
        # Progress fill
        fill_width = int(width * progress)
        pygame.draw.rect(self.screen, self.green, (x, y, fill_width, height))
        
        # Border
        pygame.draw.rect(self.screen, self.white, (x, y, width, height), 2)
    
    def draw_center_cross(self):
        """Draw fixation cross at center"""
        cross_size = 30
        center_x, center_y = self.window_size[0]//2, self.window_size[1]//2
        
        pygame.draw.line(self.screen, self.white,
                       (center_x - cross_size, center_y),
                       (center_x + cross_size, center_y), 4)
        pygame.draw.line(self.screen, self.white,
                       (center_x, center_y - cross_size),
                       (center_x, center_y + cross_size), 4)
    
    def draw_boxes(self, left_color, right_color, left_border=None, right_border=None):
        """
        Draw stimulus boxes
        
        Args:
            left_color: Color for left box
            right_color: Color for right box
            left_border: Tuple of (color, width) for left border
            right_border: Tuple of (color, width) for right border
        """
        # Default borders
        if left_border is None:
            left_border = (self.black, 3)
        if right_border is None:
            right_border = (self.black, 3)
        
        # Draw left box
        pygame.draw.rect(self.screen, left_color,
                        (self.left_pos[0], self.left_pos[1], 
                         self.box_size, self.box_size))
        pygame.draw.rect(self.screen, left_border[0],
                        (self.left_pos[0], self.left_pos[1], 
                         self.box_size, self.box_size), left_border[1])
        
        # Draw right box
        pygame.draw.rect(self.screen, right_color,
                        (self.right_pos[0], self.right_pos[1], 
                         self.box_size, self.box_size))
        pygame.draw.rect(self.screen, right_border[0],
                        (self.right_pos[0], self.right_pos[1], 
                         self.box_size, self.box_size), right_border[1])
        
        # Draw labels
        for i, (pos, label) in enumerate([(self.left_pos, self.labels[0]), 
                                          (self.right_pos, self.labels[1])]):
            text = self.font.render(label, True, self.white)
            text_rect = text.get_rect(centerx=pos[0] + self.box_size//2,
                                      bottom=pos[1] - 20)
            self.screen.blit(text, text_rect)
    
    def draw_calibration_boxes(self, target_idx, elapsed_time):
        """
        Draw boxes during calibration with flashing
        
        Args:
            target_idx: Index of target box to flash (0 or 1)
            elapsed_time: Time elapsed since calibration step started
        """
        for i, (pos, label) in enumerate([(self.left_pos, self.labels[0]),
                                          (self.right_pos, self.labels[1])]):
            if i == target_idx:
                # This box should flicker at its frequency
                freq = self.frequencies[i]
                phase = np.sin(2 * np.pi * freq * elapsed_time)
                intensity = int((phase + 1) * 127.5)
                color = (intensity, intensity, intensity)
                border_color = self.green if intensity > 127 else self.yellow
                border_width = 6
            else:
                # Non-target box - dim static
                color = (50, 50, 50)
                border_color = (100, 100, 100)
                border_width = 2
            
            pygame.draw.rect(self.screen, color,
                           (pos[0], pos[1], self.box_size, self.box_size))
            pygame.draw.rect(self.screen, border_color,
                           (pos[0], pos[1], self.box_size, self.box_size), border_width)
            
            # Label
            text_color = self.white if i == target_idx else (150, 150, 150)
            text = self.font.render(label, True, text_color)
            text_rect = text.get_rect(centerx=pos[0] + self.box_size//2,
                                      bottom=pos[1] - 20)
            self.screen.blit(text, text_rect)
    
    def draw_selection_feedback(self, selection_text, color):
        """Draw selection feedback text"""
        text = self.font.render(selection_text, True, color)
        text_rect = text.get_rect(centerx=self.window_size[0]//2,
                                  bottom=self.window_size[1] - 80)
        self.screen.blit(text, text_rect)
    
    def draw_confidence_bar(self, confidence):
        """Draw confidence bar"""
        if confidence <= 0:
            return
            
        bar_width = 300
        bar_height = 20
        bar_x = (self.window_size[0] - bar_width) // 2
        bar_y = self.window_size[1] - 50
        
        # Background
        pygame.draw.rect(self.screen, self.black,
                       (bar_x, bar_y, bar_width, bar_height))
        
        # Fill based on confidence
        fill_width = int(bar_width * min(1.0, confidence))
        if confidence > 0.7:
            bar_color = self.green
        elif confidence > 0.4:
            bar_color = self.yellow
        else:
            bar_color = self.red
        
        pygame.draw.rect(self.screen, bar_color,
                       (bar_x, bar_y, fill_width, bar_height))
        
        # Border
        pygame.draw.rect(self.screen, self.white,
                       (bar_x, bar_y, bar_width, bar_height), 2)
    
    def draw_scores(self, scores, threshold):
        """Draw frequency scores"""
        for i, (freq, score) in enumerate(zip(self.frequencies, scores)):
            text = f"{freq}Hz: {score:.2f}"
            color = self.green if score > threshold else self.white
            score_text = self.small_font.render(text, True, color)
            score_rect = score_text.get_rect(x=20, y=self.window_size[1] - 100 + i*25)
            self.screen.blit(score_text, score_rect)
    
    def draw_calibration_info(self, step, total_steps, message, elapsed, duration):
        """Draw calibration information"""
        # Clear with calibration background
        self.screen.fill((64, 64, 64))
        
        # Title
        title = f"CALIBRATION - Step {step} of {total_steps}"
        self.draw_title(title)
        
        # Instruction
        text = self.font.render(message, True, self.white)
        text_rect = text.get_rect(centerx=self.window_size[0]//2, y=150)
        self.screen.blit(text, text_rect)
        
        # Progress bar
        progress = min(1.0, elapsed / duration)
        self.draw_progress_bar(progress)
        
        # Time remaining
        remaining = max(0, duration - elapsed)
        time_text = f"Time remaining: {remaining:.1f}s"
        text = self.small_font.render(time_text, True, self.white)
        text_rect = text.get_rect(centerx=self.window_size[0]//2, y=230)
        self.screen.blit(text, text_rect)
    
    def draw_recording_indicator(self):
        """Draw recording indicator during calibration"""
        indicator_text = "● RECORDING SSVEP RESPONSE"
        text = self.small_font.render(indicator_text, True, self.red)
        text_rect = text.get_rect(centerx=self.window_size[0]//2,
                                  bottom=self.window_size[1] - 20)
        self.screen.blit(text, text_rect)
    
    def update_flicker(self, start_time):
        """
        Calculate flicker states for stimulus boxes
        
        Args:
            start_time: When the flickering started
            
        Returns:
            Tuple of (left_color, right_color)
        """
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Calculate phases
        left_phase = np.sin(2 * np.pi * self.frequencies[0] * elapsed)
        right_phase = np.sin(2 * np.pi * self.frequencies[1] * elapsed)
        
        # Convert to colors
        left_intensity = int((left_phase + 1) * 127.5)
        right_intensity = int((right_phase + 1) * 127.5)
        
        left_color = (left_intensity, left_intensity, left_intensity)
        right_color = (right_intensity, right_intensity, right_intensity)
        
        return left_color, right_color
    
    def flip(self):
        """Update the display"""
        pygame.display.flip()
    
    def tick(self, fps=60):
        """Control frame rate"""
        if self.clock:
            self.clock.tick(fps)
    
    def quit(self):
        """Clean up pygame"""
        pygame.quit()