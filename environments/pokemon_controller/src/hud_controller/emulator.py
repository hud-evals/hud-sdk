from __future__ import annotations

import base64
import io
import os
from typing import TYPE_CHECKING, Any, cast

from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .display_adapters import DisplayAdapter, NativeAdapter

if TYPE_CHECKING:
    from PIL import Image


class Emulator:
    def __init__(
        self,
        rom_path: str,
        display_adapter: DisplayAdapter | None = None,
        save_path: str | None = None,
        speed: int = 1,
    ) -> None:
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found at {rom_path}")

        window = "null"
        scale = 1
        if isinstance(display_adapter, NativeAdapter):
            window = display_adapter.window
            scale = display_adapter.scale

        self.pyboy = PyBoy(rom_path, window=window, scale=scale, no_input=True)
        self.pyboy.set_emulation_speed(speed)

        self.display_adapter = display_adapter

        if save_path and os.path.exists(save_path):
            with open(save_path, "rb") as f:
                self.pyboy.load_state(f)
        self.save_path = save_path

        self.is_paused = False

    def _encode_image(self, image: Image.Image) -> str:
        """Encodes a PIL Image object to base64."""
        buffered = io.BytesIO()
        # Convert RGBA to RGB if necessary
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_observation(self) -> dict[str, Any]:
        """Captures the current game state and returns an Observation object."""
        if not self.pyboy:
            return {"text": "Emulator not running"}

        # Capture screenshot
        screenshot = self.get_screenshot()
        screenshot_data = self._encode_image(screenshot)

        # Extract game context
        game_context = self.get_game_context()

        # Create observation object
        observation = {
            "text": str(game_context),
            "screenshot": screenshot_data,
        }
        return observation

    def get_screenshot(self) -> Image.Image:
        """Captures a screenshot of the current game screen."""
        return cast("Image.Image", self.pyboy.screen.image)

    def get_game_context(self) -> dict:
        """Extracts key game context information from memory."""
        if not self.pyboy:
            # Return default/error values if emulator isn't running
            return {
                "player_x": -1,
                "player_y": -1,
                "map_id": -1,
                "error": "Emulator not running",
            }

        try:
            # Memory addresses for Pokemon Red/Blue
            player_x = self.pyboy.memory[0xD361]
            player_y = self.pyboy.memory[0xD362]
            map_id = self.pyboy.memory[0xD35E]
            return {
                "player_x": player_x,
                "player_y": player_y,
                "map_id": map_id,
                "is_paused": self.is_paused,
            }
        except Exception as e:
            # Handle potential errors during memory access
            # Return default/error values
            return {"player_x": -1, "player_y": -1, "map_id": -1, "error": str(e)}

    def get_game_context_log(self) -> dict:
        """Extracts a wider range of game context information for logging."""
        if not self.pyboy:
            return {"error": "Emulator not running"}

        try:
            # Helper function to read multi-byte BCD for money
            def read_bcd(address: int, num_bytes: int) -> int:
                value = 0
                for i in range(num_bytes):
                    byte = self.pyboy.memory[address + i]
                    value += (byte >> 4) * (10 ** (2 * (num_bytes - 1 - i) + 1))
                    value += (byte & 0x0F) * (10 ** (2 * (num_bytes - 1 - i)))
                return value

            # Helper function to read 2-byte value (e.g., HP)
            def read_word(address: int) -> int:
                return (self.pyboy.memory[address] << 8) + self.pyboy.memory[address + 1]

            # Helper function to read 3-byte value (e.g., EXP)
            def read_3byte(address: int) -> int:
                b1 = self.pyboy.memory[address]
                b2 = self.pyboy.memory[address + 1]
                b3 = self.pyboy.memory[address + 2]
                return (b1 << 16) + (b2 << 8) + b3

            # Helper function to read IVs/DVs from two bytes
            def read_ivs(address: int) -> dict:
                byte1 = self.pyboy.memory[address]  # Attack/Defense
                byte2 = self.pyboy.memory[address + 1]  # Speed/Special
                return {
                    "attack_dv": byte1 >> 4,
                    "defense_dv": byte1 & 0x0F,
                    "speed_dv": byte2 >> 4,
                    "special_dv": byte2 & 0x0F,
                }

            # Read party data
            num_pokemon = self.pyboy.memory[0xD163]
            party_pokemon_data = []
            party_pokemon_ids = [
                self.pyboy.memory[0xD164 + i] for i in range(num_pokemon)
            ]  # D164-D169

            pokemon_struct_size = 0x2C  # Size of each pokemon data block in memory

            for i in range(num_pokemon):
                base_addr = 0xD16B + i * pokemon_struct_size
                pokemon_info = {
                    "species_id": self.pyboy.memory[
                        base_addr
                    ],  # D16B, D197, etc. (Matches D164-D169 list)
                    "hp_current": read_word(base_addr + 0x01),  # +1, +2
                    "status": self.pyboy.memory[base_addr + 0x04],  # +4
                    "type1": self.pyboy.memory[base_addr + 0x05],  # +5
                    "type2": self.pyboy.memory[base_addr + 0x06],  # +6
                    "move1_id": self.pyboy.memory[base_addr + 0x08],  # +8
                    "move2_id": self.pyboy.memory[base_addr + 0x09],  # +9
                    "move3_id": self.pyboy.memory[base_addr + 0x0A],  # +A
                    "move4_id": self.pyboy.memory[base_addr + 0x0B],  # +B
                    "trainer_id": read_word(base_addr + 0x0C),  # +C, +D
                    "exp": read_3byte(base_addr + 0x0E),  # +E, +F, +10
                    "hp_ev": read_word(base_addr + 0x11),  # +11, +12
                    "attack_ev": read_word(base_addr + 0x13),  # +13, +14
                    "defense_ev": read_word(base_addr + 0x15),  # +15, +16
                    "speed_ev": read_word(base_addr + 0x17),  # +17, +18
                    "special_ev": read_word(base_addr + 0x19),  # +19, +1A
                    "ivs": read_ivs(base_addr + 0x1B),  # +1B, +1C
                    "pp_move1": self.pyboy.memory[base_addr + 0x1D],  # +1D
                    "pp_move2": self.pyboy.memory[base_addr + 0x1E],  # +1E
                    "pp_move3": self.pyboy.memory[base_addr + 0x1F],  # +1F
                    "pp_move4": self.pyboy.memory[base_addr + 0x20],  # +20
                    "level": self.pyboy.memory[base_addr + 0x21],  # +21
                    "hp_max": read_word(base_addr + 0x22),  # +22, +23
                    "attack": read_word(base_addr + 0x24),  # +24, +25
                    "defense": read_word(base_addr + 0x26),  # +26, +27
                    "speed": read_word(base_addr + 0x28),  # +28, +29
                    "special": read_word(base_addr + 0x2A),  # +2A, +2B
                }
                party_pokemon_data.append(pokemon_info)

            context = {
                # Player/General State
                "player_x": self.pyboy.memory[0xD361],
                "player_y": self.pyboy.memory[0xD362],
                "map_id": self.pyboy.memory[0xD35E],
                "player_direction": self.pyboy.memory[0xC109],  # 0: down, 4: up, 8: left, 12: right
                "in_battle": self.pyboy.memory[0xD057],  # 0: no, 1: yes
                "money": read_bcd(0xD347, 3),
                "badges": self.pyboy.memory[0xD356],  # Bit flags
                "have_town_map": self.pyboy.memory[0xD5F3],
                "have_oaks_parcel": self.pyboy.memory[0xD60D],
                "bike_speed": self.pyboy.memory[0xD700],  # Unsure of exact meaning
                "fly_anywhere": read_word(0xD70B),  # Cheat flag?
                "safari_zone_time": read_word(0xD70E),
                "fossilized_pokemon": self.pyboy.memory[0xD710],  # Flag?
                "position_in_air": self.pyboy.memory[0xD714],  # While flying?
                "got_lapras": self.pyboy.memory[0xD72E],
                "is_ss_anne_here": self.pyboy.memory[0xD803],
                # Trainer/Pokemon Defeated Flags
                "fought_giovanni": self.pyboy.memory[0xD751],
                "fought_brock": self.pyboy.memory[0xD755],
                "fought_misty": self.pyboy.memory[0xD75E],
                "fought_lt_surge": self.pyboy.memory[0xD773],
                "fought_erika": self.pyboy.memory[0xD77C],
                "fought_articuno": self.pyboy.memory[0xD782],
                "fought_koga": self.pyboy.memory[0xD792],
                "fought_blaine": self.pyboy.memory[0xD79A],
                "fought_sabrina": self.pyboy.memory[0xD7B3],
                "fought_zapdos": self.pyboy.memory[0xD7D4],
                "fought_snorlax_vermilion": self.pyboy.memory[0xD7D8],
                "fought_snorlax_celadon": self.pyboy.memory[0xD7E0],
                "fought_moltres": self.pyboy.memory[0xD7EE],
                "fought_mewtwo": self.pyboy.memory[0xD85F],
                # Party Info
                "num_pokemon_in_party": num_pokemon,
                "party_pokemon_ids": party_pokemon_ids,  # Simple list of species IDs in order
                "party_data": party_pokemon_data,  # List of detailed dicts for each pokemon
            }
            return context
        except Exception as e:
            # Return partial context or error indicator
            return {"error": str(e)}

    def press_button(self, button_name: str) -> None:
        """Presses a specific button on the emulator."""
        button_map = {
            "UP": WindowEvent.PRESS_ARROW_UP,
            "DOWN": WindowEvent.PRESS_ARROW_DOWN,
            "LEFT": WindowEvent.PRESS_ARROW_LEFT,
            "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
            "A": WindowEvent.PRESS_BUTTON_A,
            "B": WindowEvent.PRESS_BUTTON_B,
            "START": WindowEvent.PRESS_BUTTON_START,
            "SELECT": WindowEvent.PRESS_BUTTON_SELECT,
        }
        release_map = {
            "UP": WindowEvent.RELEASE_ARROW_UP,
            "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
            "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
            "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
            "A": WindowEvent.RELEASE_BUTTON_A,
            "B": WindowEvent.RELEASE_BUTTON_B,
            "START": WindowEvent.RELEASE_BUTTON_START,
            "SELECT": WindowEvent.RELEASE_BUTTON_SELECT,
        }

        button_name = button_name.upper()

        if button_name == "PAUSE":
            self.is_paused = not self.is_paused
        elif button_name in button_map:
            press_event = button_map[button_name.upper()]
            release_event = release_map[button_name.upper()]
            self.pyboy.send_input(press_event)
            self.tick(6)  # Advance 6 frames
            self.pyboy.send_input(release_event)
            self.tick()  # Advance one frame
        else:
            pass

    def press_button_sequence(self, buttons: list[str]) -> None:
        """Presses a sequence of buttons."""
        for button in buttons:
            self.press_button(button)
            self.pyboy.tick(10)  # Advance 10 frames between presses

    def tick(self, frames: int = 1) -> bool:
        if self.is_paused:
            return True  # Skip ticks if paused

        """Advance the emulator by a number of frames."""
        for _ in range(frames):
            self.pyboy.tick()
            if self.display_adapter:
                # Update display using the adapter
                image = self.get_screenshot().convert("RGB")
                self.display_adapter.update_display(image)
                self.display_adapter.update_audio(self.pyboy.sound.ndarray.copy())
                action_list = self.display_adapter.handle_events()
                if isinstance(action_list, list):
                    for action in action_list:
                        self.press_button(action)

                if not self.display_adapter.running:
                    return False
        return True  # Indicate emulation continues

    def save_state(self) -> None:
        """Saves the current emulator state to the specified save file."""
        if self.save_path:
            with open(self.save_path, "wb") as f:
                self.pyboy.save_state(f)

    def stop(self) -> None:
        """Stops the emulator."""
        # Check if pyboy instance exists before stopping (v2 doesn't have .running)
        if hasattr(self, "pyboy"):
            self.pyboy.stop()
        else:
            pass
