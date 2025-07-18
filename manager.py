"""
Manages a pool of LLMClient instances with dynamic scaling, health checks,
and metrics tracking.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, Callable, Optional, List
from llm_client import LLMClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

class LLMManager:
    def __init__(
        self,
        model_path: str,
        initial_instances: int = 2,
        max_instances: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        device: str = 'cpu',
        health_check_interval: int = 60
    ):
        self.model_path = model_path
        self.device = device
        self._lock = threading.Lock()
        self._instances: List[LLMClient] = []
        self._executor = ThreadPoolExecutor(max_workers=max_instances)
        self._metrics: Dict[str, float] = {'requests': 0, 'errors': 0}
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.max_instances = max_instances
        self.health_check_interval = health_check_interval

        # spin up initial clients
        for _ in range(initial_instances):
            self._add_instance()

        # start health monitor
        threading.Thread(target=self._health_monitor, daemon=True).start()

    def _add_instance(self):
        client = LLMClient(self.model_path, self.device)
        self._instances.append(client)
        logger.info(f"Added new LLM client instance. Total: {len(self._instances)}")

    def _remove_instance(self):
        if self._instances:
            self._instances.pop()
            logger.info(f"Removed an LLM client instance. Total: {len(self._instances)}")

    def _health_monitor(self):
        while True:
            time.sleep(self.health_check_interval)
            with self._lock:
                load = self._metrics['requests'] > 0 and (self._metrics['errors'] / self._metrics['requests'])
                # dynamic scaling based on error rate
                if load > self.scale_up_threshold and len(self._instances) < self.max_instances:
                    self._add_instance()
                elif load < self.scale_down_threshold and len(self._instances) > 1:
                    self._remove_instance()
                # reset metrics
                self._metrics = {'requests': 0, 'errors': 0}

    def submit(
        self,
        prompt: str,
        callback: Optional[Callable[[str], None]] = None,
        **gen_kwargs
    ) -> Future:
        """
        Submit a prompt for generation. Optional callback receives the output.
        """
        with self._lock:
            self._metrics['requests'] += 1
            # round-robin assignment
            client = self._instances.pop(0)
            self._instances.append(client)

        future = self._executor.submit(self._safe_generate, client, prompt, callback, **gen_kwargs)
        return future

    def _safe_generate(self, client: LLMClient, prompt: str, callback: Optional[Callable], **gen_kwargs) -> Optional[List[str]]:
        try:
            result = client.generate(prompt, **gen_kwargs)
            if callback:
                callback(result)
            return result
        except Exception as e:
            with self._lock:
                self._metrics['errors'] += 1
            logger.error(f"Error during generation: {e}")
            return None

    def shutdown(self, wait: bool = True):
        """Gracefully shutdown all instances and thread pool."""
        logger.info("Shutting down LLMManager...")
        self._executor.shutdown(wait=wait)
        logger.info("Shutdown complete.")
