import torch
from pi_zero_pytorch import π0
import time

BATCH_SIZES = [
    2**i for i in range(0, 5)
]
print(BATCH_SIZES)

ref_vision = torch.randn(1, 1024, 512)
ref_commands = torch.randint(0, 20_000, (1, 1024))
ref_joint_state = torch.randn(1, 12)
ref_actions = torch.randn(1, 32, 6)

a = time.perf_counter()
model = π0(
    dim = 512,
    dim_action_input = 6,
    dim_joint_state = 12,
    num_tokens = 20_000
)
b = time.perf_counter()
print(f"Model initialized in {b-a:.2f} seconds")

ref_output = model(ref_vision, ref_commands, ref_joint_state, ref_actions)
abs_diffs = []

for BATCH_SIZE in BATCH_SIZES:
    print(f"Running batch size {BATCH_SIZE}")
    vision = torch.randn(BATCH_SIZE, 1024, 512)
    commands = torch.randint(0, 20_000, (BATCH_SIZE, 1024))
    joint_state = torch.randn(BATCH_SIZE, 12)
    actions = torch.randn(BATCH_SIZE, 32, 6)

    vision[0] = ref_vision
    commands[0] = ref_commands
    joint_state[0] = ref_joint_state
    actions[0] = ref_actions

    output = model(vision, commands, joint_state, actions)
    abs_diff = (output - ref_output).abs().max()
    abs_diffs.append(abs_diff)
    print(f"Absolute difference for batch size {BATCH_SIZE}: {abs_diff}")
