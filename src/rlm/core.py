"""Core RLM implementation."""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

from .types import Message
from .repl import REPLExecutor, REPLError
from .prompts import build_system_prompt
from .parser import parse_response, is_final


class RLMError(Exception):
    """Base error for RLM."""

    pass


TraceHook = Callable[[Dict[str, Any]], None]


class MaxIterationsError(RLMError):
    """Max iterations exceeded."""

    pass


class MaxDepthError(RLMError):
    """Max recursion depth exceeded."""

    pass


class BudgetExceededError(RLMError):
    """Token budget exceeded.

    By design, the call that crosses the budget is allowed to finish, and then
    RLM stops before issuing any additional LLM calls.
    """

    def __init__(
        self,
        message: str,
        *,
        token_budget: int,
        prompt_tokens: int,
        completion_tokens: int,
        last_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.token_budget = token_budget
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.last_response = last_response


@dataclass
class _SharedUsage:
    token_budget: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    last_call_prompt_tokens: int = 0
    last_call_completion_tokens: int = 0
    last_call_total_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _read_int_field(obj: Any, *names: str) -> int | None:
    for name in names:
        value = None
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_usage_counts(response: Any) -> Tuple[int, int] | None:
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt = _read_int_field(usage, "prompt_tokens", "input_tokens")
    completion = _read_int_field(usage, "completion_tokens", "output_tokens")

    if prompt is None or completion is None:
        total = _read_int_field(usage, "total_tokens")
        if total is None:
            return None
        return (int(total), 0)

    return (prompt, completion)


class RLM:
    """Recursive Language Model."""

    def __init__(
        self,
        model: str,
        recursive_model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_depth: int = 5,
        max_iterations: int = 30,
        max_total_tokens: Optional[int] = None,
        trace_hook: TraceHook | None = None,
        trace_context: Dict[str, Any] | None = None,
        _current_depth: int = 0,
        _shared_usage: Optional[_SharedUsage] = None,
        **llm_kwargs: Any
    ):
        """
        Initialize RLM.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-sonnet-4", "ollama/llama3.2")
            recursive_model: Optional cheaper model for recursive calls
            api_base: Optional API base URL
            api_key: Optional API key
            max_depth: Maximum recursion depth
            max_iterations: Maximum REPL iterations per call
            max_total_tokens: Optional total token budget across all LLM calls for this run
            trace_hook: Optional callback invoked with JSON-serializable trace events.
            trace_context: Optional dict merged into each trace event (e.g., sample_id).
            _current_depth: Internal current depth tracker
            _shared_usage: Internal shared usage tracker (used for recursion)
            **llm_kwargs: Additional LiteLLM parameters
        """
        self.model = model
        self.recursive_model = recursive_model or model
        self.api_base = api_base
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_total_tokens = max_total_tokens
        self._current_depth = _current_depth
        self._shared_usage = _shared_usage or _SharedUsage(token_budget=max_total_tokens)
        self.llm_kwargs = llm_kwargs

        self.repl = REPLExecutor()

        self._trace_hook = trace_hook
        self._trace_context = trace_context or {}

        # Stats
        self._iterations = 0

    def _trace(self, event: Dict[str, Any]) -> None:
        hook = self._trace_hook
        if hook is None:
            return
        try:
            payload = {
                **self._trace_context,
                **event,
            }
            hook(payload)
        except Exception:
            # Trace hooks must never break the core execution path.
            return

    def completion(
        self,
        query: str = "",
        context: str = "",
        **kwargs: Any
    ) -> str:
        """
        Sync wrapper for acompletion.

        Args:
            query: User query (optional if query is in context)
            context: Context to process (optional, can pass query here)
            **kwargs: Additional LiteLLM parameters

        Returns:
            Final answer string

        Examples:
            # Standard usage
            rlm.completion(query="Summarize this", context=document)

            # Query in context (RLM will extract task)
            rlm.completion(context="Summarize this document: ...")

            # Single string (treat as context)
            rlm.completion("Process this text and extract dates")
        """
        # If only one argument provided, treat it as context
        if query and not context:
            context = query
            query = ""

        return asyncio.run(self.acompletion(query, context, **kwargs))

    async def acompletion(
        self,
        query: str = "",
        context: str = "",
        **kwargs: Any
    ) -> str:
        """
        Main async completion method.

        Args:
            query: User query (optional if query is in context)
            context: Context to process (optional, can pass query here)
            **kwargs: Additional LiteLLM parameters

        Returns:
            Final answer string

        Raises:
            MaxIterationsError: If max iterations exceeded
            MaxDepthError: If max recursion depth exceeded

        Examples:
            # Explicit query and context
            await rlm.acompletion(query="What is this?", context=doc)

            # Query embedded in context
            await rlm.acompletion(context="Extract all dates from: ...")

            # LLM will figure out the task
            await rlm.acompletion(context=document_with_instructions)
        """
        # If only query provided, treat it as context
        if query and not context:
            context = query
            query = ""
        if self._current_depth >= self.max_depth:
            raise MaxDepthError(f"Max recursion depth ({self.max_depth}) exceeded")

        # Initialize REPL environment
        repl_env = self._build_repl_env(query, context)

        # Build initial messages
        system_prompt = build_system_prompt(len(context), self._current_depth)
        messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Main loop
        for iteration in range(self.max_iterations):
            if (
                self.max_total_tokens is not None
                and self._shared_usage.total_tokens >= int(self.max_total_tokens)
            ):
                self._trace(
                    {
                        "event_type": "token_budget_exceeded",
                        "iteration": iteration + 1,
                        "depth": self._current_depth,
                        "token_budget": int(self.max_total_tokens),
                        "tokens_spent": int(self._shared_usage.total_tokens),
                    }
                )
                raise BudgetExceededError(
                    f"Token budget exceeded: {self._shared_usage.total_tokens} >= {self.max_total_tokens}",
                    token_budget=int(self.max_total_tokens),
                    prompt_tokens=self._shared_usage.prompt_tokens,
                    completion_tokens=self._shared_usage.completion_tokens,
                    last_response=None,
                )

            self._iterations = iteration + 1

            # Call LLM
            response = await self._call_llm(messages, iteration=iteration + 1, **kwargs)

            # Check for FINAL
            if is_final(response):
                answer = parse_response(response, repl_env)
                if answer is not None:
                    return answer

            if (
                self.max_total_tokens is not None
                and self._shared_usage.total_tokens >= int(self.max_total_tokens)
            ):
                raise BudgetExceededError(
                    f"Token budget exceeded: {self._shared_usage.total_tokens} >= {self.max_total_tokens}",
                    token_budget=int(self.max_total_tokens),
                    prompt_tokens=self._shared_usage.prompt_tokens,
                    completion_tokens=self._shared_usage.completion_tokens,
                    last_response=response,
                )

            # Execute code in REPL
            try:
                exec_result = self.repl.execute(response, repl_env)
            except REPLError as e:
                exec_result = f"Error: {str(e)}"
            except Exception as e:
                exec_result = f"Unexpected error: {str(e)}"

            self._trace(
                {
                    "event_type": "repl_exec",
                    "iteration": iteration + 1,
                    "depth": self._current_depth,
                    "assistant_response": response,
                    "exec_result": exec_result,
                }
            )

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": exec_result})

        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without FINAL()"
        )

    async def _call_llm(
        self,
        messages: List[Message],
        *,
        iteration: int | None = None,
        **kwargs: Any
    ) -> str:
        """
        Call LLM API.

        Args:
            messages: Conversation messages
            **kwargs: Additional parameters (can override model here)

        Returns:
            LLM response text
        """
        self._shared_usage.llm_calls += 1

        # Choose model based on depth
        default_model = self.model if self._current_depth == 0 else self.recursive_model

        # Allow override via kwargs
        model = kwargs.pop('model', default_model)

        # Merge kwargs
        call_kwargs = {**self.llm_kwargs, **kwargs}
        if self.api_base:
            call_kwargs['api_base'] = self.api_base
        if self.api_key:
            call_kwargs['api_key'] = self.api_key

        # Call LiteLLM
        start_time = time.perf_counter()
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            **call_kwargs
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        usage = _extract_usage_counts(response)
        if usage is not None:
            prompt_tokens, completion_tokens = usage
            self._shared_usage.prompt_tokens += int(prompt_tokens)
            self._shared_usage.completion_tokens += int(completion_tokens)
            self._shared_usage.last_call_prompt_tokens = int(prompt_tokens)
            self._shared_usage.last_call_completion_tokens = int(completion_tokens)
            self._shared_usage.last_call_total_tokens = int(prompt_tokens) + int(completion_tokens)
        elif self.max_total_tokens is not None:
            raise RLMError(
                "Provider did not return token usage; cannot enforce max_total_tokens budget."
            )

        # Extract text
        text = response.choices[0].message.content

        self._trace(
            {
                "event_type": "llm_call",
                "iteration": iteration,
                "depth": self._current_depth,
                "model": model,
                "messages": messages,
                "response": text,
                "latency_ms": latency_ms,
                "prompt_tokens": int(self._shared_usage.last_call_prompt_tokens),
                "completion_tokens": int(self._shared_usage.last_call_completion_tokens),
                "total_tokens": int(self._shared_usage.last_call_total_tokens),
                "tokens_spent": int(self._shared_usage.total_tokens),
                "token_budget": int(self.max_total_tokens) if self.max_total_tokens is not None else None,
            }
        )
        return text

    def _build_repl_env(self, query: str, context: str) -> Dict[str, Any]:
        """
        Build REPL environment.

        Args:
            query: User query
            context: Context string

        Returns:
            Environment dict
        """
        env: Dict[str, Any] = {
            'context': context,
            'query': query,
            'recursive_llm': self._make_recursive_fn(),
            're': re,  # Whitelist re module
        }
        return env

    def _make_recursive_fn(self) -> Any:
        """
        Create recursive LLM function for REPL.

        Returns:
            Async function that can be called from REPL
        """
        async def recursive_llm(sub_query: str, sub_context: str) -> str:
            """
            Recursively process sub-context.

            Args:
                sub_query: Query for sub-context
                sub_context: Sub-context to process

            Returns:
                Answer from recursive call
            """
            if self._current_depth + 1 >= self.max_depth:
                return f"Max recursion depth ({self.max_depth}) reached"

            # Create sub-RLM with increased depth
            sub_rlm = RLM(
                model=self.recursive_model,
                recursive_model=self.recursive_model,
                api_base=self.api_base,
                api_key=self.api_key,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                max_total_tokens=self.max_total_tokens,
                trace_hook=self._trace_hook,
                trace_context=self._trace_context,
                _current_depth=self._current_depth + 1,
                _shared_usage=self._shared_usage,
                **self.llm_kwargs
            )

            return await sub_rlm.acompletion(sub_query, sub_context)

        # Wrap in sync function for REPL compatibility
        def sync_recursive_llm(sub_query: str, sub_context: str) -> str:
            """Sync wrapper for recursive_llm."""
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in async context, but REPL is sync
                # Create a new thread to run async code
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        recursive_llm(sub_query, sub_context)
                    )
                    return future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                return asyncio.run(recursive_llm(sub_query, sub_context))

        return sync_recursive_llm

    @property
    def stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return {
            'llm_calls': self._shared_usage.llm_calls,
            'iterations': self._iterations,
            'depth': self._current_depth,
            'prompt_tokens': self._shared_usage.prompt_tokens,
            'completion_tokens': self._shared_usage.completion_tokens,
            'total_tokens': self._shared_usage.total_tokens,
        }
