import torch
from typing import List
from torch._dynamo.utils import CompileProfiler


__all__ = ["my_compiler","jit_backend","compare_prof"]

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                full_graph: bool = False,backend: str = "inductor",
                options: dict ={'trace.graph_diagram': True,'trace.enabled':True}):
    """
    This is a custom compiler for FX graphs. It is called by the
    `torch.fx.GraphModule` class when the `compile` method is called.
    """
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

def jit_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    scripted = torch.jit.script(gm)
    return torch.jit.jit_backend(scripted)

def compare_prof(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor],logger=None):
    prof= CompileProfiler()
    prof_mode = torch.compile(gm,backend=prof)
    prof_mode(example_inputs)
    if logger:
        logger.info(prof.report())
    print(prof.report())