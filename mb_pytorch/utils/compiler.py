"""Utilities for PyTorch model compilation and optimization."""

from typing import List, Dict, Any, Optional, Union, Callable
import torch
import torch.fx
from torch._dynamo.utils import CompileProfiler
from torch.jit.mobile import _load_for_lite_interpreter
import logging
import warnings
from pathlib import Path

__all__ = ["CompilerUtils", "TorchScriptUtils"]

class CompilerUtils:
    """Utilities for compiling PyTorch models with various backends."""
    
    SUPPORTED_BACKENDS = ['inductor', 'eager', 'aot_eager', 'cudagraphs', 'nvfuser']
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize compiler utilities.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
        
    def compile_model(
        self,
        model: torch.nn.Module,
        example_inputs: Union[torch.Tensor, List[torch.Tensor]],
        backend: str = "inductor",
        options: Optional[Dict[str, Any]] = None,
        full_graph: bool = True
    ) -> torch.nn.Module:
        """
        Compile model with specified backend.
        
        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing
            backend: Compilation backend
            options: Compilation options
            full_graph: Whether to compile full graph
            
        Returns:
            Compiled model
            
        Raises:
            ValueError: If backend is not supported
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Backend {backend} not supported. "
                f"Supported backends: {self.SUPPORTED_BACKENDS}"
            )
            
        if options is None:
            options = {
                'trace.graph_diagram': True,
                'trace.enabled': True,
                'trace.debug_log': True
            }
            
        try:
            if self.logger:
                self.logger.info(f"Compiling model with {backend} backend")
                
            compiled_model = torch.compile(
                model,
                backend=backend,
                options=options,
                fullgraph=full_graph
            )
            
            # Test compilation with example inputs
            if isinstance(example_inputs, torch.Tensor):
                example_inputs = [example_inputs]
            _ = compiled_model(*example_inputs)
            
            if self.logger:
                self.logger.info("Model compilation successful")
                
            return compiled_model
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Model compilation failed: {str(e)}")
            raise
            
    def profile_compilation(
        self,
        model: torch.nn.Module,
        example_inputs: Union[torch.Tensor, List[torch.Tensor]],
        num_warmup: int = 5,
        num_iter: int = 100
    ) -> Dict[str, Any]:
        """
        Profile model compilation and execution.
        
        Args:
            model: PyTorch model to profile
            example_inputs: Example inputs for profiling
            num_warmup: Number of warmup iterations
            num_iter: Number of profiling iterations
            
        Returns:
            Profiling results
        """
        if self.logger:
            self.logger.info("Starting compilation profiling")
            
        profiler = CompileProfiler()
        compiled_model = torch.compile(model, backend=profiler)
        
        if isinstance(example_inputs, torch.Tensor):
            example_inputs = [example_inputs]
            
        # Warmup
        for _ in range(num_warmup):
            _ = compiled_model(*example_inputs)
            
        # Profile iterations
        for _ in range(num_iter):
            _ = compiled_model(*example_inputs)
            
        results = profiler.report()
        
        if self.logger:
            self.logger.info("Profiling completed")
            self.logger.info(results)
            
        return results
    
    @staticmethod
    def optimize_for_inference(
        model: torch.nn.Module,
        example_inputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Optimized model
        """
        model.eval()
        if isinstance(example_inputs, torch.Tensor):
            example_inputs = [example_inputs]
            
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)
            traced_model = torch.jit.freeze(traced_model)
            
        return traced_model


class TorchScriptUtils:
    """Utilities for TorchScript model handling."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize TorchScript utilities.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger
        
    def save_torchscript(
        self,
        model: torch.nn.Module,
        save_path: Union[str, Path],
        example_inputs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        use_trace: bool = True,
        optimize: bool = True
    ) -> None:
        """
        Save model as TorchScript.
        
        Args:
            model: PyTorch model to save
            save_path: Path to save model
            example_inputs: Example inputs for tracing
            use_trace: Whether to use tracing (vs scripting)
            optimize: Whether to optimize for inference
            
        Raises:
            ValueError: If trace requires example_inputs but none provided
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        
        if use_trace:
            if example_inputs is None:
                raise ValueError("Tracing requires example_inputs")
                
            if isinstance(example_inputs, torch.Tensor):
                example_inputs = [example_inputs]
                
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_inputs)
                if optimize:
                    traced_model = torch.jit.freeze(traced_model)
                traced_model.save(str(save_path))
        else:
            scripted_model = torch.jit.script(model)
            if optimize:
                scripted_model = torch.jit.freeze(scripted_model)
            scripted_model.save(str(save_path))
            
        if self.logger:
            self.logger.info(f"Model saved to {save_path}")
            
    def load_torchscript(
        self,
        model_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        optimize_for_mobile: bool = False
    ) -> torch.jit.ScriptModule:
        """
        Load TorchScript model.
        
        Args:
            model_path: Path to model file
            map_location: Device to load model to
            optimize_for_mobile: Whether to optimize for mobile
            
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            if optimize_for_mobile:
                model = _load_for_lite_interpreter(str(model_path))
            else:
                model = torch.jit.load(str(model_path), map_location=map_location)
                
            if self.logger:
                self.logger.info(f"Model loaded from {model_path}")
                
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def optimize_for_mobile(
        self,
        model: torch.nn.Module,
        example_inputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.jit.ScriptModule:
        """
        Optimize model for mobile deployment.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Optimized model
        """
        model.eval()
        if isinstance(example_inputs, torch.Tensor):
            example_inputs = [example_inputs]
            
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_inputs)
            traced_model = torch.jit.freeze(traced_model)
            
            # Additional mobile optimizations
            traced_model.eval()
            traced_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
        return traced_model
    
    def set_to_max_cores(self) -> None:
        """Set TorchScript to use all available cores."""
        print('Setting TorchScript to use all available cores')
        print(f'Current number of threads for torch: {torch.get_num_threads()}')
        print(f'Number of available cores: {torch.get_num_interop_threads()}')
        torch.set_num_threads(torch.get_num_interop_threads()-2)
        cores = torch.get_num_interop_threads()-2
        print(f'Number of threads set to: {cores}. Using 2 less than available cores for other tasks.')