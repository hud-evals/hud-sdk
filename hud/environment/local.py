"""LocalEnvironment implementation using Docker exec."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Optional

import docker
from docker.errors import DockerException, ImageNotFound

from .base import Environment, EvaluateConfig, SetupConfig, process_config

logger = logging.getLogger("hud.environment")

# Define the directory where local environments are stored
ENVIRONMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "environments"
)

def check_docker_status() -> bool:
    """Check if Docker daemon is running and responsive.
    
    Returns:
        bool: True if Docker is running, False otherwise
    """
    try:
        # Try to connect to Docker
        client = docker.from_env()
        client.ping()
        return True
    except DockerException:
        return False

def _build_and_run_docker(env_id: str, mount_code: bool = True) -> str:
    """Build and run a Docker container for a local environment.
    
    Args:
        env_id: Environment ID to build
        mount_code: Whether to mount the code volume (faster) or copy later
        
    Returns:
        str: Container ID
    """
    env_path = os.path.join(ENVIRONMENTS_DIR, env_id)
    dockerfile_path = os.path.join(env_path, "Dockerfile")
    
    if not os.path.exists(dockerfile_path):
        raise ValueError(f"Dockerfile not found for environment {env_id}")
    
    # Enable BuildKit for faster builds
    os.environ["DOCKER_BUILDKIT"] = "1"
    
    # Build the Docker image
    image_name = f"hud-{env_id}"
    logger.info("Building Docker image %s", image_name)
    
    try:
        # Create Docker client
        client = docker.from_env()
        
        # Check if Docker is running
        if not check_docker_status():
            raise RuntimeError("Docker daemon not running")
        
        # Check if image already exists
        try:
            client.images.get(image_name)
            logger.info("Using existing Docker image %s", image_name)
        except ImageNotFound:
            # Build image if it doesn't exist
            logger.debug("Building Docker image %s from %s", image_name, env_path)
            client.images.build(path=env_path, tag=image_name, rm=True)
            logger.info("Docker build completed successfully")
    except Exception as e:
        logger.error("Docker build failed: %s", e)
        raise
    
    # Run the Docker container
    container_name = f"hud-{env_id}-{uuid.uuid4()}"
    
    logger.info("Running Docker container %s", container_name)
    try:
        volumes = {}
        if mount_code:
            abs_env_path = os.path.abspath(os.path.join(ENVIRONMENTS_DIR, env_id))
            volumes[abs_env_path] = {"bind": "/environment", "mode": "rw"}
            logger.debug("Mounting code directory: %s -> /environment", abs_env_path)
        
        container = client.containers.run(
            image_name,
            name=container_name,
            detach=True,
            volumes=volumes if mount_code else None
        )
        
        container_id = container.id
        
        if not container_id:
            raise RuntimeError("Failed to get container ID")
            
        logger.info("Container %s started", container_id)
        return container_id
    except Exception as e:
        logger.error("Docker run failed: %s", e)
        raise


class LocalEnvironment(Environment):
    """Local Docker-based environment.
    
    This environment runs in a local Docker container and executes commands
    using Docker exec. It automatically mounts the environment code and
    handles package installation from pyproject.toml.
    """
    
    def __init__(
        self,
        id: str,
        metadata: Optional[dict[str, Any]] = None,
        mount_code: bool = True,
    ) -> None:
        """Initialize a local environment.
        
        Args:
            id: The environment ID
            metadata: Optional metadata
            mount_code: Whether to mount code directory (default: True)
        """
        self.id = id
        self.metadata = metadata or {}
        self.container_id = None
        self.mount_code = mount_code
        self.url = None
        self.live_url = None
        
        # For preloaded setup and evaluate configurations
        self._preloaded_setup = None
        self._preloaded_evaluate = None
        
    def preload_setup(self, setup_config: SetupConfig) -> None:
        """Preload setup configuration from a Task.
        
        Args:
            setup_config: The setup configuration
        """
        logger.debug("Preloading setup configuration: %s", setup_config)
        self._preloaded_setup = setup_config
        
    def preload_evaluate(self, evaluate_config: EvaluateConfig) -> None:
        """Preload evaluation configuration from a Task.
        
        Args:
            evaluate_config: The evaluation configuration
        """
        logger.debug("Preloading evaluate configuration: %s", evaluate_config)
        self._preloaded_evaluate = evaluate_config
    
    async def create_environment(self) -> None:
        """Create and initialize the environment.
        
        This method launches the Docker container and initializes the environment.
        """
        logger.debug("Starting create_environment() method")
        
        # Check if Docker is running before proceeding
        if not check_docker_status():
            raise RuntimeError("Docker is not running or not responding")
            
        # Always create a new container
        loop = asyncio.get_event_loop()
        self.container_id = await loop.run_in_executor(
            None,
            lambda: _build_and_run_docker(self.id, mount_code=self.mount_code)
        )
        if self.container_id is None:
            raise RuntimeError("Failed to create container")
        logger.info("LocalEnvironment container created: %s", self.container_id)
        
        # Initialize the environment
        try:
            # Set a timeout for the initialize method
            await asyncio.wait_for(self._initialize(), timeout=30)
            logger.info("Local environment %s initialized", self.id)
        except asyncio.TimeoutError as e:
            logger.error("Environment initialization timed out after 30 seconds")
            # Try to clean up
            try:
                # Use run_in_executor to avoid blocking the event loop
                client = docker.from_env()
                container_id = self.container_id  # Keep a reference to the container_id
                
                # Run the cleanup in an executor to avoid blocking
                await loop.run_in_executor(
                    None, 
                    lambda: self._cleanup_container(client, container_id)
                )
            except Exception as cleanup_e:
                logger.error("Failed to clean up container: %s", cleanup_e)
            self.container_id = None
            raise TimeoutError("Environment initialization timed out") from e
        
        logger.debug("create_environment() method completed successfully")
        
    def _cleanup_container(self, client: docker.DockerClient, container_id: str) -> None:
        """Helper method to clean up a Docker container.
        
        Args:
            client: Docker client
            container_id: ID of the container to clean up
        """
        try:
            container = client.containers.get(container_id)
            container.stop()
            container.remove()
            logger.info("Cleaned up container %s", container_id)
        except docker.errors.NotFound:
            logger.warning("Container %s not found during cleanup", container_id)
        except Exception as e:
            logger.error("Error during container cleanup: %s", e)
            # Don't re-raise, we're already handling another exception
    
    async def _initialize(self) -> None:
        """Initialize the environment.
        
        This method verifies that the Docker container is running,
        checks that Docker exec is working correctly, and installs
        packages from pyproject.toml if available.
        """
        logger.debug("Starting initialize() method")
        
        # Create Docker client
        client = docker.from_env()
        
        # Check if container is running
        try:
            container = client.containers.get(self.container_id)
            if container.status != "running":
                logger.error("Container %s is not running", self.container_id)
                raise RuntimeError(f"Container {self.container_id} is not running")
            logger.debug("Container %s is running", self.container_id)
        except docker.errors.NotFound as e:
            logger.error("Container %s not found", self.container_id)
            raise RuntimeError(f"Container {self.container_id} not found: {e}")
        
        # Small test command to verify exec works
        try:
            exec_result = container.exec_run("echo Container is ready")
            if exec_result.exit_code != 0:
                raise RuntimeError(f"Test command failed with code {exec_result.exit_code}")
            logger.debug("Test command result: %s", exec_result.output.decode().strip())
        except Exception as e:
            logger.error("Failed to run test command: %s", e)
            raise RuntimeError(f"Failed to run test command: {e}")
            
        # Install packages from pyproject.toml
        logger.info("Installing packages from pyproject.toml")
        try:
            install_result = await self.execute("cd /environment && pip install -q -e .")
            if install_result.get("exit_code", 1) != 0:
                logger.warning("Package installation failed: %s", install_result.get('stderr', ''))
                logger.warning("Continuing anyway, as some environments might not have a pyproject.toml")
            else:
                logger.info("Package installation completed successfully")
        except Exception as e:
            logger.warning("Error during package installation: %s", e)
            logger.warning("Continuing despite installation error")
            
        logger.debug("initialize() method completed successfully")
    
    async def setup(self, setup_config: Optional[SetupConfig] = None) -> Any:
        """Run a setup function in the environment.
        
        Args:
            setup_config: The setup configuration to run
            
        Returns:
            Any: Result of the setup function
        """
        # If no config provided and we have preloaded config, use that
        if setup_config is None and self._preloaded_setup is not None:
            setup_config = self._preloaded_setup
        elif setup_config is None:
            raise ValueError("No setup configuration provided and no preloaded setup configuration")
        
        logger.debug("Processing setup configuration: %s", setup_config)
        processed_configs = process_config(setup_config)
        
        # Handle empty configs
        if not processed_configs:
            logger.warning("Empty setup configuration")
            return []
        
        # Handle multiple configs
        if len(processed_configs) > 1:
            results = []
            for config in processed_configs:
                func_name = f"setup.{config['function']}"
                result = await self._run_function(func_name, *config["args"])
                results.append(result)
            return results
        
        # Handle single config
        config = processed_configs[0]
        func_name = f"setup.{config['function']}"
        return await self._run_function(func_name, *config["args"])
        
    async def step(self, command: str) -> Any:
        """Execute a step in the environment.
        
        Args:
            command: The command to execute
            
        Returns:
            Any: Result of the step execution
        """
        logger.debug("Executing step command: %s", command)
        return await self._run_function("step.step", command)
        
    async def evaluate(self, evaluate_config: Optional[EvaluateConfig] = None) -> Any:
        """Run an evaluation function in the environment.
        
        Args:
            evaluate_config: The evaluation configuration to run
            
        Returns:
            Any: Result of the evaluation function
        """
        # If no config provided and we have preloaded config, use that
        if evaluate_config is None and self._preloaded_evaluate is not None:
            evaluate_config = self._preloaded_evaluate
        elif evaluate_config is None:
            raise ValueError("No evaluation configuration provided and no preloaded evaluation configuration")
        
        logger.debug("Processing evaluation configuration: %s", evaluate_config)
        processed_configs = process_config(evaluate_config)
        
        # Handle empty configs
        if not processed_configs:
            logger.warning("Empty evaluation configuration")
            return []
        
        # Handle multiple configs
        if len(processed_configs) > 1:
            results = []
            for config in processed_configs:
                func_name = f"evaluate.{config['function']}"
                result = await self._run_function(func_name, *config["args"])
                results.append(result)
            return results
        
        # Handle single config
        config = processed_configs[0]
        func_name = f"evaluate.{config['function']}"
        return await self._run_function(func_name, *config["args"])
        
    async def get_info(self, function_name: str = "get_state", *args: Any, **kwargs: Any) -> Any:
        """Get information from the environment.
        
        Args:
            function_name: The name of the info function to run (default: "get_state")
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the info function
        """
        logger.debug("Getting info with function: %s", function_name)
        return await self._run_function(f"info.{function_name}", *args, **kwargs)
        
    async def get_urls(self) -> dict[str, str]:
        """Get URLs for the environment.
        
        Returns:
            dict: Dictionary of URLs for accessing the environment
        """
        try:
            return await self.get_info("get_urls")
        except Exception as e:
            logger.error("Failed to get URLs: %s", e)
            return {}
    
    async def execute(self, command: str) -> dict[str, Any]:
        """Execute a command in the local environment using Docker exec.
        
        Args:
            command: The command to execute
            
        Returns:
            dict: Results with stdout, stderr, and exit_code
        """
        logger.debug("Executing command via Docker exec: %s", command)
        try:
            # Get Docker client and container
            client = docker.from_env()
            container = client.containers.get(self.container_id)
            
            # Run the command with Docker exec
            exec_result = container.exec_run(
                cmd=["sh", "-c", command],
                demux=True  # Split stdout and stderr
            )
            
            # Handle the response format
            if isinstance(exec_result.output, tuple) and len(exec_result.output) == 2:
                stdout, stderr = exec_result.output
                stdout = stdout.decode() if stdout else ""
                stderr = stderr.decode() if stderr else ""
            else:
                stdout = exec_result.output.decode() if isinstance(exec_result.output, bytes) else exec_result.output
                stderr = ""
            
            exit_code = exec_result.exit_code
            
            logger.debug("Command completed with exit code: %d", exit_code)
            logger.debug("stdout length: %d, stderr length: %d", len(stdout), len(stderr))
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }
        except Exception as e:
            logger.error("Command execution failed: %s", e)
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1
            }
    
    async def _run_function(
        self,
        function_path: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Run a function in the environment (internal helper).
        
        Args:
            function_path: Path to the function as "module.function"
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the function call
        """
        module_name, func_name = function_path.split(".")
        
        # Use base64 encoding to avoid escaping issues
        import base64
        
        # Serialize arguments to JSON then encode
        args_b64 = base64.b64encode(json.dumps(args).encode()).decode()
        kwargs_b64 = base64.b64encode(json.dumps(kwargs).encode()).decode()
        
        # Construct Python command with the encoded arguments and improved error handling
        cmd = f"""python -c "
import json, sys, base64, traceback, os

# Add environment directory to Python path
sys.path.insert(0, '/environment')

try:
    # Try direct import first (for modules in root of environment directory)
    try:
        from {module_name} import {func_name}
    except ImportError:
        # If that fails, list what modules are available (for debugging)
        print(os.listdir('/environment'), file=sys.stderr)
        print(sys.path, file=sys.stderr)
        raise
        
    args = json.loads(base64.b64decode('{args_b64}').decode())
    kwargs = json.loads(base64.b64decode('{kwargs_b64}').decode())
    
    # Print debug info to help with debugging
    print(f'\\n### DEBUG INFO ###\\nModule: {module_name}\\nFunction: {func_name}\\nArgs: {{args}}\\nKwargs: {{kwargs}}\\n', file=sys.stderr)
    
    result = {func_name}(*args, **kwargs)
    print(json.dumps(result))
    sys.exit(0)
except ImportError as e:
    error_info = {{
        'error': str(e),
        'traceback': traceback.format_exc(),
        'type': 'import_error',
        'details': f'Could not import {{module_name}}.{{func_name}}. Check that the module exists and function is defined.',
        'module': '{module_name}',
        'function': '{func_name}'
    }}
    print(json.dumps(error_info))
    sys.exit(1)
except Exception as e:
    error_info = {{
        'error': str(e),
        'traceback': traceback.format_exc(),
        'type': 'execution_error',
        'details': f'Error executing {{module_name}}.{{func_name}}: {{e}}',
        'module': '{module_name}',
        'function': '{func_name}'
    }}
    print(json.dumps(error_info))
    sys.exit(1)
"
"""
        logger.debug("Executing function: %s.%s", module_name, func_name)
        result = await self.execute(cmd)
        
        if result.get("exit_code", 1) != 0:
            # Log detailed error info
            logger.error("Function execution failed: %s.%s", module_name, func_name)
            logger.error("Exit code: %s", result.get("exit_code"))
            logger.error("stderr: %s", result.get("stderr", ""))
            
            # Try to parse error JSON from stdout or return a formatted error
            try:
                error_data = json.loads(result.get("stdout", "{}"))
                if isinstance(error_data, dict) and "error" in error_data:
                    # This is an error returned from our script
                    logger.error("Error details: %s", error_data.get("details", "No details"))
                    logger.error("Traceback: %s", error_data.get("traceback", "No traceback"))
                    return error_data
                else:
                    # Unexpected output format
                    return {
                        "error": "Unexpected function output format",
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "exit_code": result.get("exit_code", 1)
                    }
            except json.JSONDecodeError:
                # Failed to parse JSON, return raw output
                return {
                    "error": f"Error in {module_name}.{func_name} (could not parse error details)",
                    "raw_stdout": result.get("stdout", ""),
                    "raw_stderr": result.get("stderr", ""),
                    "exit_code": result.get("exit_code", 1)
                }
            
        try:
            return json.loads(result.get("stdout", "{}"))
        except json.JSONDecodeError:
            return {
                "error": f"Failed to parse result from {module_name}.{func_name}",
                "raw": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
        
    async def close(self) -> None:
        """Stop and remove the Docker container."""
        if not self.container_id:
            logger.warning("No container ID to close")
            return
            
        logger.info("Closing environment, stopping container %s", self.container_id)
        try:
            # Get Docker client and container
            client = docker.from_env()
            try:
                container = client.containers.get(self.container_id)
                container.stop()
                container.remove()
                logger.info("Removed container %s", self.container_id)
            except docker.errors.NotFound:
                logger.warning("Container %s not found, may have been already removed", self.container_id)
        except Exception as e:
            logger.error("Failed to remove container: %s", e)
            raise
    async def wait_for_ready(self) -> None:
        """Wait for the environment to be ready."""
        return

