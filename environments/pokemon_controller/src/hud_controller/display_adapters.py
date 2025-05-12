from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image


class DisplayAdapter:
    def __init__(self, muted: bool = False) -> None:
        self.muted = muted
        self.running = True

    def update_display(self, image: Image.Image) -> None:
        """Update the display with the given image."""

    def update_audio(self, audio: np.ndarray) -> None:
        """Update the audio with the given audio data."""

    def handle_events(self) -> list[str] | None:
        """Handle events for the display."""


class PygameAdapter(DisplayAdapter):
    def __init__(
        self,
        screen_size: tuple[int, int] = (480, 432),
        muted: bool = False,
        handle_input: bool = False,
    ) -> None:
        super().__init__(muted=muted)

        import pygame

        self.pygame = pygame
        self.pygame.init()
        self.mixer = pygame.mixer
        self.mixer.init(frequency=2)
        self.channel = self.mixer.Channel(0)
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("PyBoy + Pygame Display")

        self.handle_input = handle_input

    def update_display(self, image: Image.Image) -> None:
        """Update the display with the given image."""
        if self.screen is None:
            screen_size = (image.width, image.height)
            self.screen = self.pygame.display.set_mode(screen_size)
            self.pygame.display.set_caption("PyBoy + Pygame Display")

        scaled_image = image.resize(self.screen_size, resample=Image.Resampling.NEAREST)
        size = scaled_image.size
        data = scaled_image.tobytes()
        surface = self.pygame.image.fromstring(data, size, "RGB")
        self.screen.blit(surface, (0, 0))
        self.pygame.display.flip()

    def update_audio(self, audio: np.ndarray) -> None:
        if self.muted:
            return

        if audio is None or len(audio) == 0:
            return

        if audio.dtype == np.int8:
            audio = np.clip(audio.astype(np.int16) * 256, -32768, 32767)

        if audio.ndim == 2:
            audio = audio.reshape(-1)  # Flatten stereo to interleaved format

        sound = self.mixer.Sound(buffer=audio.tobytes())

        if not self.channel.get_busy():
            self.channel.play(sound)
        else:
            self.channel.queue(sound)  # Queue next chunk smoothly

    def handle_events(self) -> list[str]:
        """Handle events for the display."""
        action_list = []

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False

            if self.handle_input and event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_UP:
                    action_list.append("UP")
                elif event.key == self.pygame.K_DOWN:
                    action_list.append("DOWN")
                elif event.key == self.pygame.K_LEFT:
                    action_list.append("LEFT")
                elif event.key == self.pygame.K_RIGHT:
                    action_list.append("RIGHT")
                elif event.key == self.pygame.K_a:
                    action_list.append("A")
                elif event.key == self.pygame.K_b:
                    action_list.append("B")
                elif event.key == self.pygame.K_z:
                    action_list.append("START")
                elif event.key == self.pygame.K_x:
                    action_list.append("SELECT")

        return action_list


class NativeAdapter(DisplayAdapter):
    def __init__(self, scale: int = 3, window: Literal["SDL2", "OpenGL", "null"] = "SDL2") -> None:
        super().__init__()
        self.window = window
        self.scale = scale
