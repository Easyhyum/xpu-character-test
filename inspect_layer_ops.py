#!/usr/bin/env python3
"""
Inspect first and last transformer layer structure and operations for models.

This script loads a model using the project's `model_load_function` and attempts
to locate the transformer's layer container (common attribute names are tried).

It prints:
 - The attribute path found for the layer container
 - Number of layers
 - For the first and last layer: child submodules (name + type), forward signature,
   first part of forward source (if available), and parameter shapes/dtypes.

Optionally, with --run-dry-forward it will load a tokenizer and run a tiny
forward pass while registering lightweight hooks on the inspected layer's
submodules to record which modules are invoked during a forward call. This is
best-effort and may fail for some custom models; it's disabled by default.

Usage examples:
  python tools/inspect_layer_ops.py --from-config
  python tools/inspect_layer_ops.py --model facebook/opt-125m --run-dry-forward
"""
import argparse
import importlib
import inspect
import json
import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import torch

# Import the local model loader
try:
    from api.model_loader import model_load_function
except Exception as e:
    print(f"Failed to import model_load_function from api.model_loader: {e}")
    raise

COMMON_LAYER_PATH_CANDIDATES = [
    "model.layers",
    "model.model.layers",
    "transformer.h",
    "transformer.blocks",
    "model.transformer.h",
    "model.encoder.layers",
    "model.decoder.layers",
    "blocks",
    "layers",
    "encoder.layer",
    "encoder.layers",
]


def find_layer_container(model) -> Tuple[Optional[object], Optional[str]]:
    """Try to find the main transformer layer container and return (obj, path).

    Returns (None, None) if nothing found.
    """
    for cand in COMMON_LAYER_PATH_CANDIDATES:
        obj = model
        ok = True
        for part in cand.split('.'):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj, cand

    # fallback: search for the largest ModuleList in named_modules
    from torch.nn import ModuleList

    best = None
    best_name = None
    for name, mod in model.named_modules():
        if isinstance(mod, ModuleList) and len(mod) > 0:
            if best is None or len(mod) > len(best):
                best = mod
                best_name = name
    if best is not None:
        return best, best_name

    # final fallback: look for attributes that look like lists of layers
    for name, mod in model.named_modules():
        # if its immediate children are numbered keys (0,1,2...) or many children
        try:
            children = list(getattr(model, name).children())
        except Exception:
            continue
        if len(children) >= 2:
            return getattr(model, name), name

    return None, None


def print_layer_info(layer, title: str):
    print(f"\n--- {title} ---")
    print(f"Type: {type(layer)}")
    # children
    print("Children:")
    for n, m in layer.named_children():
        print(f"  - {n}: {type(m)}")

    # forward signature & source
    if hasattr(layer, 'forward'):
        try:
            sig = inspect.signature(layer.forward)
            print(f"forward signature: {sig}")
        except Exception as e:
            print(f"forward signature: <unavailable> ({e})")

        try:
            src = inspect.getsource(layer.forward)
            print("forward source (first 1k chars):")
            print(src[:1000])
        except Exception as e:
            print(f"forward source: <unavailable or builtin> ({e})")

    # parameters
    print("Parameters:")
    params = list(layer.named_parameters())
    for n, p in params:
        print(f"  - {n}: shape={tuple(p.shape)} dtype={p.dtype} numel={p.numel()}")

    # Helper: get submodule by dotted path inside this layer
    def get_submodule(root, path: str):
        if not path:
            return None
        obj = root
        for part in path.split('.'):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    # Find activation modules and forward-source activation names for heuristics
    act_modules = []
    for n, m in layer.named_modules():
        if n == "":
            continue
        has_params = any(True for _ in m.parameters())
        tname = type(m).__name__.lower()
        if not has_params and any(k in tname for k in ("silu", "gelu", "relu", "glu", "swiglu", "swi")):
            act_modules.append((n, type(m).__name__))

    forward_activation_names = []
    try:
        if hasattr(layer, 'forward'):
            src = inspect.getsource(layer.forward)
            import re
            forward_activation_names = re.findall(r"\b(silu|gelu|relu|glu|swiglu|silu_new|gelu_new)\b", src, flags=re.IGNORECASE)
            forward_activation_names = list(dict.fromkeys([m.lower() for m in forward_activation_names]))
    except Exception:
        forward_activation_names = []

    # Map each parameter to likely activation (heuristic): if param belongs to mlp, map to mlp.act_fn or src-detected
    print("\nParameter -> likely activation (heuristic):")
    for n, p in params:
        module_path = '.'.join(n.split('.')[:-1])
        parent = get_submodule(layer, module_path)
        likely_act = None
        # prefer explicit act module inside layer (e.g., mlp.act_fn)
        for aname, atype in act_modules:
            if module_path.startswith(aname.split('.')[0]):
                likely_act = aname
                break
        # fallback to forward source detected names
        if likely_act is None and forward_activation_names:
            # if parameter is inside mlp, prefer forward activation name
            if module_path.startswith('mlp') or 'gate' in n or 'up_proj' in n or 'down_proj' in n:
                likely_act = ','.join(forward_activation_names)

        # if parent is Linear, show in/out features
        linear_info = None
        try:
            import torch.nn as nn
            if isinstance(parent, nn.Linear):
                try:
                    linear_info = (getattr(parent, 'in_features', None), getattr(parent, 'out_features', None))
                except Exception:
                    linear_info = None
        except Exception:
            linear_info = None

        print(f"  - {n}: activation={likely_act or 'unknown'}{(' | linear(in,out)='+str(linear_info)) if linear_info else ''}")

    # Detect activation modules (modules without parameters like SiLU, GELU, ReLU)
    act_modules = []
    for n, m in layer.named_modules():
        # skip the top-level module itself
        if n == "":
            continue
        # module has no parameters -> candidate activation or functional-only module
        has_params = any(True for _ in m.parameters())
        if not has_params:
            tname = type(m).__name__.lower()
            # look for common activation module names
            if any(k in tname for k in ("silu", "gelu", "relu", "glu", "swiglu", "gelu")):
                act_modules.append((n, type(m)))

    if act_modules:
        print("Detected activation modules (no parameters):")
        for n, t in act_modules:
            print(f"  - {n}: {t}")
    else:
        print("Detected activation modules: none found as parameterless nn.Modules (will check forward source)")

    # Try to detect activation function calls inside forward source
    try:
        if hasattr(layer, 'forward'):
            src = inspect.getsource(layer.forward)
            import re
            matches = re.findall(r"\b(silu|gelu|relu|glu|swiglu|swi_g?lu|gelu_new)\b", src, flags=re.IGNORECASE)
            matches = list(dict.fromkeys([m.lower() for m in matches]))
            if matches:
                print(f"Activation names found in forward source: {matches}")
            else:
                print("No obvious activation function names found in forward source (search by regex).")
    except Exception:
        pass


def run_dry_forward_capture(model, tokenizer, device, sample_length=8):
    """Run a tiny forward pass to capture which submodules of first/last layer are called.

    Best-effort; some models may require extra kwargs. We attempt a simple input_ids
    forward which works for most AutoModelForCausalLM / AutoModel models.
    Returns a dict mapping module -> list(call_count).
    """
    layer_container, path = find_layer_container(model)
    if layer_container is None:
        print("Cannot locate layer container for dry forward capture.")
        return {}

    # pick first and last layer
    try:
        first = layer_container[0]
        last = layer_container[-1]
    except Exception:
        print("Layer container is not indexable - skipping dry forward capture.")
        return {}

    # Build hook list for first/last layer named modules
    modules_to_hook = []
    for n, m in first.named_modules():
        modules_to_hook.append((f"first.{n}", m))
    for n, m in last.named_modules():
        modules_to_hook.append((f"last.{n}", m))

    # We'll record ordered events (sequence of module calls) with details
    events = []
    handles = []

    def capture_tensor_shapes(obj):
        try:
            if isinstance(obj, torch.Tensor):
                return tuple(obj.shape)
            if isinstance(obj, (list, tuple)):
                return [capture_tensor_shapes(x) for x in obj]
            return None
        except Exception:
            return None

    def make_hook(name, module):
        def hook(module_, input, output):
            # capture whether module has parameters and a short list of param names
            try:
                param_names = [n for n, _ in module.named_parameters()]
            except Exception:
                param_names = []
            has_params = len(param_names) > 0
            in_shapes = capture_tensor_shapes(input)
            out_shapes = capture_tensor_shapes(output)
            events.append({
                "module": name,
                "type": type(module).__name__,
                "has_params": has_params,
                "param_names": param_names[:10],
                "input_shapes": in_shapes,
                "output_shapes": out_shapes,
            })
        return hook

    for name, mod in modules_to_hook:
        try:
            h = mod.register_forward_hook(make_hook(name, mod))
            handles.append(h)
        except Exception:
            # some modules don't support hooks
            pass

    # prepare input
    vocab_size = getattr(tokenizer, 'vocab_size', None) or getattr(model.config, 'vocab_size', None) or 1000
    device = device or torch.device('cpu')
    input_ids = torch.randint(low=0, high=max(2, int(vocab_size)), size=(1, sample_length), dtype=torch.long, device=device)

    model_device = None
    try:
        # move model to device for forward
        params = list(model.parameters())
        if len(params) > 0:
            model_device = params[0].device
        else:
            model_device = torch.device('cpu')
        model.to(device)
    except Exception:
        model_device = device

    try:
        with torch.no_grad():
            # Try likely forward kwargs
            kwargs = {"input_ids": input_ids}
            try:
                model(**kwargs)
            except Exception as e:
                # try with attention_mask
                kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
                model(**kwargs)
    except Exception as e:
        print(f"Dry forward failed: {e}")
    finally:
        # remove hooks
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        # try to move model back to original device
        try:
            model.to(model_device)
        except Exception:
            pass

    # return ordered events (may contain duplicates if module called multiple times)
    # limit size to avoid huge outputs
    max_events = 1000
    return events[:max_events]


def inspect_model(model_name: str, run_dry_forward: bool = False, summarize_ops: bool = False):
    print(f"\n=== Inspecting model: {model_name} ===")
    tokenizer = None
    try:
        tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: failed to load tokenizer for {model_name}: {e}")

    model = model_load_function(model_name)
    if model is None:
        print(f"Model loader returned None for {model_name}")
        return

    layer_container, path = find_layer_container(model)
    if layer_container is None:
        print("Could not locate transformer layer container. Listing top-level children instead:")
        for n, m in model.named_children():
            print(f"  - {n}: {type(m)}")
        return

    try:
        num_layers = len(layer_container)
    except Exception:
        num_layers = None

    print(f"Found layer container at path: {path}")
    print(f"Layer container type: {type(layer_container)}")
    print(f"Reported number of layers: {num_layers}")

    # Inspect first and last layer
    try:
        first = layer_container[0]
        last = layer_container[-1]
    except Exception as e:
        print(f"Layer container not indexable: {e}")
        return

    print_layer_info(first, "First layer (index 0)")
    print_layer_info(last, "Last layer (index -1)")

    if run_dry_forward:
        print("\nRunning dry forward to capture invoked submodules (best-effort)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            called = run_dry_forward_capture(model, tokenizer, device)
            events = called
            print("Modules invoked during dry forward (ordered events):")
            if events:
                for i, ev in enumerate(events):
                    print(f"{i:03d}: {ev['module']} | {ev['type']} | has_params={ev['has_params']} | params={ev.get('param_names')} | in={ev.get('input_shapes')} -> out={ev.get('output_shapes')}")
            else:
                print("  (no hooked modules were invoked or capture failed)")
        except Exception as e:
            print(f"Dry forward capture failed: {e}")
        # Summarize into high-level steps if requested
        try:
            if events and summarize_ops:
                print("\nSummarized operation steps (best-effort):")
                # try to get forward source from first inspected layer for heuristics
                forward_src = ""
                try:
                    forward_src = inspect.getsource(model.__class__.forward)
                except Exception:
                    forward_src = ""
                steps = summarize_events_to_steps(events, forward_src)
                for s in steps:
                    print(s)
        except Exception as e:
            print(f"Failed to summarize events: {e}")


def load_models_from_config(path="hyperparameter.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg.get('models', [])


def summarize_events_to_steps(events, forward_src: str = ""):
    """Best-effort map of low-level module call events to human-friendly op steps.

    Returns a list of strings describing the sequence of operations.
    """
    steps = []
    i = 0
    n = len(events)

    def is_attention_module_name(name):
        return any(x in name for x in ("self_attn", "attn", "attention"))

    while i < n:
        ev = events[i]
        mname = ev['module']
        lname = mname.lower()

        # QKV detection: look for q_proj/k_proj/v_proj under same attn parent
        if ('q_proj' in lname or 'k_proj' in lname or 'v_proj' in lname) and is_attention_module_name(lname):
            # collect subsequent q/k/v from same parent
            parent = '.'.join(mname.split('.')[:-1])
            qkv = {'q': None, 'k': None, 'v': None}
            j = i
            while j < n:
                evj = events[j]
                if not evj['module'].startswith(parent):
                    break
                mn = evj['module'].lower()
                if 'q_proj' in mn and qkv['q'] is None:
                    qkv['q'] = evj
                if 'k_proj' in mn and qkv['k'] is None:
                    qkv['k'] = evj
                if 'v_proj' in mn and qkv['v'] is None:
                    qkv['v'] = evj
                j += 1
            # print grouped QKV step
            step_lines = ["QKV Initialize (separate projections):"]
            for key in ('q', 'k', 'v'):
                if qkv[key] is not None:
                    evq = qkv[key]
                    step_lines.append(f" - Input X -> {evq['module']} (Linear) | in={evq.get('input_shapes')} -> out={evq.get('output_shapes')} | params={evq.get('param_names')}")
            steps.append('\n'.join(step_lines))
            i = j
            continue

        # QK^T and attention internals: softmax may be functional; try to detect by forward_src or following ops
        if is_attention_module_name(lname) and ('o_proj' in lname or lname.endswith('self_attn') or '.self_attn.' in lname):
            # heuristics: after QKV, attention may compute QK, softmax, attn*V, output projection
            parent = '.'.join(mname.split('.')[:-1])
            # find o_proj in subsequent events
            j = i
            found_softmax = False
            found_o = False
            while j < n and events[j]['module'].startswith(parent):
                mn = events[j]['module'].lower()
                if 'softmax' in (events[j].get('type','').lower()) or 'softmax' in (events[j].get('param_names') or []):
                    found_softmax = True
                if 'o_proj' in mn:
                    found_o = True
                j += 1
            # Add generic attention steps
            att_lines = ["Attention block operations:"]
            att_lines.append(" 1. Compute Q, K, V via separate linear projections (see QKV Initialize)")
            att_lines.append(" 2. Compute attention scores: Q @ K^T / sqrt(d_k)")
            if 'softmax' in forward_src.lower() or found_softmax:
                att_lines.append(" 3. Apply softmax to attention scores")
            else:
                att_lines.append(" 3. (softmax – may be functional; not visible as module)")
            att_lines.append(" 4. Multiply attention weights by V -> context")
            if found_o:
                att_lines.append(" 5. Output projection: context -> output via o_proj")
            steps.append('\n'.join(att_lines))
            i = j
            continue

        # MLP detection
        if 'mlp' in lname or 'up_proj' in lname or 'down_proj' in lname or 'gate_proj' in lname:
            mlp_lines = ["MLP block operations:"]
            mlp_lines.append(" 1. gate_proj: linear (X W_gate)")
            mlp_lines.append(" 2. up_proj: linear (X W_up)")
            mlp_lines.append(" 3. Apply activation (e.g., SiLU/GELU) elementwise")
            mlp_lines.append(" 4. down_proj: linear (project back W_down)")
            steps.append('\n'.join(mlp_lines))
            i += 3
            continue

        # LayerNorms and residual adds
        if 'input_layernorm' in lname or 'post_attention_layernorm' in lname or 'rmsnorm' in lname or 'layernorm' in lname:
            steps.append(f"LayerNorm or normalization op: {mname} (applies normalization to hidden states)")
            i += 1
            continue

        # Default: list linear-like ops
        if ev.get('has_params') and ('linear' in ev.get('type','').lower() or any(x in lname for x in ('proj', 'weight'))):
            steps.append(f"Linear op: {mname} ({ev.get('type')}) params={ev.get('param_names')} in={ev.get('input_shapes')} -> out={ev.get('output_shapes')}")
            i += 1
            continue

        # otherwise just record module call
        steps.append(f"Call: {mname} ({ev.get('type')})")
        i += 1

    # Post-process: add numbering
    numbered = []
    for idx, s in enumerate(steps, start=1):
        numbered.append(f"{idx}. {s}")
    return numbered



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Single model name to inspect')
    parser.add_argument('--from-config', dest='from_config', action='store_true', help='Load models list from hyperparameter.json')
    parser.add_argument('--run-dry-forward', dest='run_dry', action='store_true', help='Run a tiny forward pass to capture module call order')
    parser.add_argument('--print-param-order', dest='print_param_order', action='store_true', help='Print a summarized param-operation order after dry forward')
    args = parser.parse_args()
    # Default behavior: if --model is provided, inspect that single model.
    # Otherwise automatically load all models from hyperparameter.json and run
    # the full inspection pipeline (dry forward + summarized param-operation print).
    models = []
    if args.model:
        models = [args.model]
        run_dry = True if args.run_dry else True  # still run dry forward by default for explicit model
        summarize_ops = True if args.print_param_order else True
        print(f"Inspecting single model: {args.model} (dry-forward + summarize enabled)")
    else:
        try:
            models = load_models_from_config()
            if not models:
                print("No models found in hyperparameter.json")
                sys.exit(1)
            print(f"Loaded {len(models)} models from hyperparameter.json; running full inspection for each (dry-forward + summarize)")
        except Exception as e:
            print(f"Failed to load models from hyperparameter.json: {e}")
            sys.exit(1)

        # Always enable dry forward and summarization in the auto-run mode
        run_dry = True
        summarize_ops = True

    for m in models:
        try:
            inspect_model(m, run_dry_forward=run_dry, summarize_ops=summarize_ops)
        except Exception as e:
            print(f"Error inspecting model {m}: {e}")


if __name__ == '__main__':
    main()
