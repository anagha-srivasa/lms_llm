"""
Orchestrates multi-instance LLM calls, validates I/O via LangGraph, supports
async batch processing, retry logic, and detailed tracing.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from llm_manager import LLMManager
from langgraph import Graph, Node, Edge

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

class OrchestrationError(Exception):
    pass

class LLMOrchestrator:
    def __init__(
        self,
        model_path: str,
        num_instances: int = 4,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.manager = LLMManager(model_path, initial_instances=num_instances)
        self.graph = Graph()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def verify_input(self, prompt: str) -> bool:
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Empty or invalid prompt detected.")
            return False
        # Additional domain-specific checks could go here
        return True

    def verify_output(self, output: Any) -> bool:
        if not output or not isinstance(output, list) or not all(isinstance(o, str) for o in output):
            logger.error("Output verification failed: expected non-empty list of strings.")
            return False
        return True

    async def _call_with_retries(
        self,
        prompt: str,
        **gen_kwargs
    ) -> List[str]:
        attempts = 0
        while attempts <= self.max_retries:
            future = self.manager.submit(prompt, **gen_kwargs)
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            if result and self.verify_output(result):
                return result
            attempts += 1
            logger.warning(f"Retry {attempts}/{self.max_retries} for prompt.")
            await asyncio.sleep(self.retry_delay)
        raise OrchestrationError(f"Failed after {self.max_retries} retries.")

    async def run_batch(
        self,
        prompts: Dict[str, str],
        **gen_kwargs
    ) -> Dict[str, List[str]]:
        """
        Run multiple prompts concurrently with validation and graph logging.
        """
        tasks: List[asyncio.Task] = []
        for key, prompt in prompts.items():
            if not self.verify_input(prompt):
                raise ValueError(f"Invalid input for key: {key}")
            # Log input node
            self.graph.add_node(Node(name=f"input:{key}", data=prompt))
            tasks.append(
                asyncio.create_task(self._call_with_retries(prompt, **gen_kwargs))
            )

        results: Dict[str, List[str]] = {}
        for key, task in zip(prompts.keys(), tasks):
            responses = await task
            # Log output node and edge
            self.graph.add_node(Node(name=f"output:{key}", data=responses))
            self.graph.add_edge(Edge(src=f"input:{key}", dst=f"output:{key}"))
            results[key] = responses
            logger.info(f"Completed orchestration for {key}.")
        return results

    def shutdown(self):
        """Cleanly shutdown manager and persist graph."""
        self.manager.shutdown()
        # Optionally serialize the graph to disk
        try:
            self.graph.to_json_file('orchestration_graph.json')
            logger.info("Graph persisted to orchestration_graph.json")
        except Exception as e:
            logger.error(f"Failed to persist graph: {e}")
