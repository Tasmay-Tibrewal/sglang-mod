# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
import builtins
import inspect
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch

try:
    from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinConfig,
        AWQMoEMethod,
    )
    # commented out vllm's bits and bytes config class and created a copy below.
    # from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
    # added the requirements for class creation directly here.
    from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
        UnquantizedLinearMethod,
        set_weight_attrs
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsW8A8Fp8MoEMethod,
        CompressedTensorsWNA16MoEMethod,
    )
    from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
    from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
    from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig
    from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )
    from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
        GPTQMarlin24Config,
    )
    from vllm.model_executor.layers.quantization.marlin import MarlinConfig
    from vllm.model_executor.layers.quantization.qqq import QQQConfig
    from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    # Define empty classes as placeholders when vllm is not available
    class DummyConfig:
        def override_quantization_method(self, *args, **kwargs):
            return None

    AQLMConfig = AWQMarlinConfig = BitsAndBytesConfig = CompressedTensorsConfig = (
        DeepSpeedFPConfig
    ) = ExpertsInt8Config = FBGEMMFp8Config = GGUFConfig = GPTQMarlin24Config = (
        MarlinConfig
    ) = QQQConfig = Int8TpuConfig = DummyConfig

# Created custom class here to handle the assert equality check for 4bit bnb, between param_data shape and weight_data shape.
# Kept everything else exactly same just changed the param empty tensor initialisation step to a correct shape, for 4 bit loading (modified just BitsAndBytesLinearMethod -> create_weights() -> create_qweight_for_4bit()).
# Class below was adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/vllm/model_executor/layers/quantization/bitsandbytes.py
class BitsAndBytesConfig(QuantizationConfig):
    """Config class for BitsAndBytes Quantization.

    Reference: https://arxiv.org/abs/2305.14314
    """

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float32",
        bnb_4bit_quant_storage: str = "uint8",
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_skip_modules: Optional[List[str]] = None,
        llm_int8_threshold: float = 6.0,
    ) -> None:

        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.llm_int8_skip_modules = llm_int8_skip_modules or []
        self.llm_int8_threshold = llm_int8_threshold

        if self.bnb_4bit_quant_storage not in ["uint8"]:
            raise ValueError("Unsupported bnb_4bit_quant_storage: "
                             f"{self.bnb_4bit_quant_storage}")

    def __repr__(self) -> str:
        return (f"BitsAndBytesConfig(load_in_8bit={self.load_in_8bit}, "
                f"load_in_4bit={self.load_in_4bit}, "
                f"bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype}, "
                f"bnb_4bit_quant_storage={self.bnb_4bit_quant_storage}, "
                f"bnb_4bit_quant_type={self.bnb_4bit_quant_type}, "
                f"llm_int8_skip_modules={self.llm_int8_skip_modules})")

    @classmethod
    def get_name(self) -> str:
        return "bitsandbytes"

    @classmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "adapter_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitsAndBytesConfig":

        def get_safe_value(config, keys, default_value=None):
            try:
                value = cls.get_from_keys(config, keys)
                return value if value is not None else default_value
            except ValueError:
                return default_value

        load_in_8bit = get_safe_value(config, ["load_in_8bit"],
                                      default_value=False)
        load_in_4bit = get_safe_value(config, ["load_in_4bit"],
                                      default_value=True)
        bnb_4bit_compute_dtype = get_safe_value(config,
                                                ["bnb_4bit_compute_dtype"],
                                                default_value="float32")
        bnb_4bit_quant_storage = get_safe_value(config,
                                                ["bnb_4bit_quant_storage"],
                                                default_value="uint8")
        bnb_4bit_quant_type = get_safe_value(config, ["bnb_4bit_quant_type"],
                                             default_value="fp4")
        bnb_4bit_use_double_quant = get_safe_value(
            config, ["bnb_4bit_use_double_quant"], default_value=False)
        llm_int8_enable_fp32_cpu_offload = get_safe_value(
            config, ["llm_int8_enable_fp32_cpu_offload"], default_value=False)
        llm_int8_has_fp16_weight = get_safe_value(config,
                                                  ["llm_int8_has_fp16_weight"],
                                                  default_value=False)
        llm_int8_skip_modules = get_safe_value(config,
                                               ["llm_int8_skip_modules"],
                                               default_value=[])
        llm_int8_threshold = get_safe_value(config, ["llm_int8_threshold"],
                                            default_value=6.0)

        return cls(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_storage=bnb_4bit_quant_storage,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_skip_modules=llm_int8_skip_modules,
            llm_int8_threshold=llm_int8_threshold)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_bnb(prefix, self.llm_int8_skip_modules):
                return UnquantizedLinearMethod()
            return BitsAndBytesLinearMethod(self)
        return None


def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: List[str]):
    # Split the prefix into its dot-separated components
    components = prefix.split('.')

    # Check if any of the skip modules exactly matches any component
    return any(module_name in components
               for module_name in llm_int8_skip_modules)

class BitsAndBytesLinearMethod(LinearMethodBase):
    """Linear method for BitsAndBytes.

    Args:
       quant_config: The BitsAndBytes quantization config.
    """

    def __init__(self, quant_config: BitsAndBytesConfig):
        try:
            import bitsandbytes
            if bitsandbytes.__version__ < "0.45.0":
                raise ImportError("bitsandbytes version is wrong. Please "
                                  "install bitsandbytes>=0.45.0.")
        except ImportError as err:
            raise ImportError("Please install bitsandbytes>=0.45.0 via "
                              "`pip install bitsandbytes>=0.45.0` to use "
                              "bitsandbytes quantizer.") from err

        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        from bitsandbytes.nn import Int8Params

        def calculate_quant_ratio(dtype):
            if dtype.is_floating_point:
                return torch.finfo(dtype).bits // torch.iinfo(torch.uint8).bits
            else:
                return torch.iinfo(dtype).bits // torch.iinfo(torch.uint8).bits

        def create_qweight_for_8bit():
            qweight = Int8Params(
                data=torch.empty(sum(output_partition_sizes),
                                 input_size_per_partition,
                                 dtype=torch.int8),
                has_fp16_weights=self.quant_config.llm_int8_has_fp16_weight,
                requires_grad=False)
            set_weight_attrs(
                qweight, {
                    "input_dim": 0,
                    "output_dim": 0,
                    "pack_factor": 1,
                    "use_bitsandbytes_8bit": True,
                    "generation": 0
                })
            return qweight

        def create_qweight_for_4bit():
            #  This was the older code in vllm library, which is replaced by a simple tensor initialisation for param.
            # quant_ratio = calculate_quant_ratio(params_dtype)
            
            # total_size = input_size_per_partition * sum(output_partition_sizes)
            # if total_size % quant_ratio != 0:
            #     raise ValueError(
            #         "The input size is not aligned with the quantized "
            #         "weight shape.")
            
            # qweight = torch.nn.Parameter(torch.empty(total_size // quant_ratio,
            #                                          1,
            #                                          dtype=torch.uint8),
            #                              requires_grad=False)

            qweight = torch.nn.Parameter(torch.empty(sum(output_partition_sizes),
                                            input_size_per_partition,
                                            dtype=torch.uint8),
                    requires_grad=False)
            set_weight_attrs(
                qweight, {
                    "input_dim": 0,
                    "output_dim": 0,
                    # "pack_factor": quant_ratio,
                    "pack_factor": 1, # (pack factor is temporarily set to 1 to fix the bug, change it later accordingly)
                    "use_bitsandbytes_4bit": True
                })
            return qweight

        if self.quant_config.load_in_8bit:
            qweight = create_qweight_for_8bit()
        else:
            qweight = create_qweight_for_4bit()
        # Enable parameters to have the same name as in the BNB
        # checkpoint format.
        layer.register_parameter("weight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.quant_config.load_in_8bit:
            return self._apply_8bit_weight(layer, x, bias)
        else:
            return self._apply_4bit_weight(layer, x, bias)

    def _apply_8bit_weight(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # only load the bitsandbytes module when needed
        from bitsandbytes import MatmulLtState, matmul

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True
        bf_x = x.to(torch.bfloat16)

        qweight = layer.weight
        offsets = qweight.bnb_shard_offsets
        quant_states = qweight.bnb_quant_state
        matmul_states = qweight.matmul_state
        generation = qweight.generation

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()])
        out = torch.empty(out_dim_0,
                          out_dim_1,
                          dtype=torch.float16,
                          device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]

            # in profile_run or the first generation of inference,
            # create new matmul_states
            if generation == 0 or generation == 1:
                matmul_states[i] = MatmulLtState()
                matmul_states[i].CB = qweight[offsets[i]:offsets[i + 1]]
                matmul_states[i].SCB = quant_states[i].to(x.device)
                matmul_states[i].threshold = (
                    self.quant_config.llm_int8_threshold)
                matmul_states[i].has_fp16_weights = (
                    self.quant_config.llm_int8_has_fp16_weight)
                matmul_states[i].is_training = False
                if matmul_states[i].threshold > 0.0 and not matmul_states[
                        i].has_fp16_weights:
                    matmul_states[i].use_pool = True

            new_x = bf_x.unsqueeze(0)

            out[:, current_index:current_index + output_size] = matmul(
                new_x,
                qweight[offsets[i]:offsets[i + 1]],
                state=matmul_states[i])

            current_index += output_size

            # only update the matmul_states if it is not profile_run
            if (generation > 0
                    and not self.quant_config.llm_int8_has_fp16_weight
                    and matmul_states[i].CB is not None
                    and matmul_states[i].CxB is not None):
                del matmul_states[i].CB
                qweight[offsets[i]:offsets[i + 1]] = matmul_states[i].CxB

        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias

        qweight.generation += 1

        return out

    def _apply_4bit_weight(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # only load the bitsandbytes module when needed
        from bitsandbytes import matmul_4bit

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True
        bf_x = x.to(torch.bfloat16)

        qweight = layer.weight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()])
        out = torch.empty(out_dim_0,
                          out_dim_1,
                          dtype=torch.bfloat16,
                          device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]
            # It is more efficient to use out kwarg like
            # matmul_4bit(..., out = ...).  Infeasible now due to the bug
            # https://github.com/TimDettmers/bitsandbytes/issues/1235.
            # Need to change  after the bug is fixed.
            out[:, current_index:current_index + output_size] = matmul_4bit(
                bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])

            current_index += output_size

        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias

        return out

from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

# Base quantization methods that don't depend on vllm
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "compressed-tensors": CompressedTensorsConfig,
}

# VLLM-dependent quantization methods
VLLM_QUANTIZATION_METHODS = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "tpu_int8": Int8TpuConfig,
    "fbgemm_fp8": FBGEMMFp8Config,
    "marlin": MarlinConfig,
    "gguf": GGUFConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
    "experts_int8": ExpertsInt8Config,
    "gptq_marlin": GPTQMarlinConfig,
    "gptq": GPTQConfig,
}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS, **VLLM_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    if quantization in VLLM_QUANTIZATION_METHODS and not VLLM_AVAILABLE:
        raise ValueError(
            f"{quantization} quantization requires some operators from vllm. "
            "Pleaes install vllm by `pip install vllm==0.7.2`"
        )

    return QUANTIZATION_METHODS[quantization]


# Match dynamic rules with module name (prefix) and override quantize
# config if module (prefix) matches a rule
def override_config(config: QuantizationConfig, prefix: str):
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    config.pack_factor = 32 // config.weight_bits  # packed into int32
    if config.get_name() == "gptq_marlin":
        is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
        if isinstance(is_sym, bool):
            config.is_sym = is_sym

        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: "
                f"bits={config.weight_bits}, sym={config.is_sym}"
            )

        config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]
    elif config.get_name() == "gptq":
        if config.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {config.weight_bits} bits."
            )


def get_dynamic_override(
    config: QuantizationConfig,
    layer_name: str,
    key: Optional[str] = None,
    default_value: Union[int, bool, None] = None,
) -> Union[Dict, int, bool, None]:
    for pattern, pattern_dict in config.dynamic.items():
        # Negative match: matched modules are excluded from quantized init
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), layer_name):
                return False
        # Positive match: matched modules have quant properties overrides
        # base quant config
        elif re.match(pattern.removeprefix("+:"), layer_name):
            if key is None:
                return pattern_dict
            else:
                return pattern_dict.get(key, default_value)
    return default_value


def get_linear_quant_method(
    config: QuantizationConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):
    cloned_config = deepcopy(config)
    parallel_lm_head_quantized = (
        isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized
    )

    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        # False = skip module, None = no override, else = Positive match
        if (
            get_dynamic_override(  # noqa: E712
                cloned_config, layer_name=prefix  # noqa: E712
            )
            == False
        ):  # noqa: E712
            if parallel_lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)

        return linear_method_cls(cloned_config)
    return None


def gptq_get_quant_method(self, layer, prefix):
    if isinstance(layer, FusedMoE):
        return GPTQMarlinMoEMethod(self)

    if isinstance(self, GPTQConfig):
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQLinearMethod
        )
    elif isinstance(self, GPTQMarlinConfig):
        return get_linear_quant_method(
            self, layer, prefix=prefix, linear_method_cls=GPTQMarlinLinearMethod
        )
    return None


original_isinstance = builtins.isinstance


def monkey_patch_isinstance_for_vllm_base_layer(reverse: bool = False):
    """
    Patch isinstance so that the `get_quant_method` in vllm's QuantizationConfig
    can recognize sglang layers
    """
    if not VLLM_AVAILABLE:
        return

    if reverse:
        builtins.isinstance = original_isinstance
        return

    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from sglang.srt.layers.linear import LinearBase as PatchedLinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as PatchedFusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding as PatchedVocabParallelEmbedding,
    )

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        if classinfo is FusedMoE:
            return original_isinstance(obj, PatchedFusedMoE)
        if classinfo is VocabParallelEmbedding:
            return original_isinstance(obj, PatchedVocabParallelEmbedding)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


def monkey_patch_moe_apply(class_obj: "FusedMoEMethodBase"):
    """
    Monkey patch the apply function of vllm's FusedMoEMethodBase.
    Convert sglang arguments to vllm arguments.
    """
    original_apply = class_obj.apply
    sig = inspect.signature(original_apply)
    param_names = list(sig.parameters.keys())
    has_correction_bias = "e_score_correction_bias" in param_names

    def new_apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
    ):
        assert activation == "silu"
        assert inplace and not no_combine

        kwargs = {
            "self": self,
            "layer": layer,
            "x": x,
            "router_logits": router_logits,
            "top_k": top_k,
            "renormalize": renormalize,
            "use_grouped_topk": use_grouped_topk,
            "topk_group": topk_group,
            "num_expert_group": num_expert_group,
            "custom_routing_function": custom_routing_function,
        }
        if correction_bias is not None:
            if not has_correction_bias:
                raise ValueError(
                    "Please increase the version of your vllm. Try `pip install vllm==0.7.2`"
                )
            kwargs["e_score_correction_bias"] = correction_bias
        return original_apply(**kwargs)

    setattr(class_obj, "apply", new_apply)


def monkey_patch_quant_configs():
    """Apply all monkey patches in one place."""
    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(GPTQConfig, "get_quant_method", gptq_get_quant_method)

    monkey_patch_moe_apply(AWQMoEMethod)
    monkey_patch_moe_apply(GPTQMarlinMoEMethod)
    monkey_patch_moe_apply(CompressedTensorsW8A8Fp8MoEMethod)
    monkey_patch_moe_apply(CompressedTensorsWNA16MoEMethod)


# Only apply monkey patches if vllm is available
if VLLM_AVAILABLE:
    monkey_patch_quant_configs()
