# SIMPLER campaign blocker

SIMPLER 0.0.1 and its pinned ManiSkill2-real2sim revision were installed under
Python 3.10 with SAPIEN 2.2.2. Environment creation was attempted for
`google_robot_pick_horizontal_coke_can`, initialization 0, but failed before
`env.reset`.

The container exposes the H100 for CUDA compute, but its NVIDIA driver
capabilities are only `compute,utility`; no NVIDIA Vulkan ICD or graphics
libraries are mounted. SAPIEN's native renderer reports that Vulkan is
incompatible and then segfaults. A second attempt installed Mesa's CPU Vulkan
driver and explicitly selected `lvp_icd.json`; SAPIEN then failed with
`vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`.

Consequently the 100-episode replay collection and 800 paired closed-loop
episodes cannot start on this container. No episode results were fabricated.
The campaign scripts are present and resumable. They require a restarted
container whose NVIDIA driver capabilities include `graphics` and whose
NVIDIA Vulkan ICD is mounted.
