# Observed native predicate progress

All values come from actual post-control-step diagnostic rows. First recorded step 1 is not a reset measurement. A close predicate true at the first observation and later false can be normal opening; ever-lost does not imply failure. No inferred grasp stages or root causes are assigned.

| Model | Task | Native atom | First satisfied steps, init 0–9 (null = never observed true) | Ever lost / 10 | Final true / 10 |
|---|---|---|---|---:|---:|
| openpi | libero_10/2 | turnon flat_stove_1 | 73, 100, 97, 88, 79, 75, 79, 78, 77, 86 | 0/10 | 10/10 |
| openpi | libero_10/2 | on moka_pot_1 flat_stove_1_cook_region | 247, 259, 239, 259, 245, 228, 264, null, 234, 231 | 0/10 | 9/10 |
| openpi | libero_10/3 | close white_cabinet_1_bottom_region | 219, 212, 249, 220, 236, 205, 214, 219, 232, 311 | 0/10 | 10/10 |
| openpi | libero_10/3 | in akita_black_bowl_1 white_cabinet_1_bottom_region | 147, 138, 144, 147, 157, 129, 139, 138, 152, 188 | 0/10 | 10/10 |
| openpi | libero_10/4 | on porcelain_mug_1 plate_1 | 109, 98, 99, 101, 130, 100, 100, 113, 104, 105 | 4/10 | 10/10 |
| openpi | libero_10/4 | on white_yellow_mug_1 plate_2 | 234, 215, 213, 226, 275, 266, 231, 237, 223, 232 | 0/10 | 10/10 |
| openpi | libero_10/9 | in white_yellow_mug_1 microwave_1_heating_region | 144, 132, 142, 153, 142, 178, 139, 134, 155, 180 | 0/10 | 10/10 |
| openpi | libero_10/9 | close microwave_1 | 241, 241, 255, 258, 278, 290, 242, 238, 249, 310 | 0/10 | 10/10 |
| openpi | libero_spatial/0 | on akita_black_bowl_1 plate_1 | 78, 104, 73, 76, 74, 67, 74, 79, 83, 81 | 0/10 | 10/10 |
| openpi | libero_spatial/1 | on akita_black_bowl_1 plate_1 | 109, 106, 101, 105, 106, 105, 105, 107, 105, 105 | 0/10 | 10/10 |
| openvla_oft | libero_10/2 | turnon flat_stove_1 | 83, 92, 91, 95, 87, 82, 86, 80, 90, 87 | 0/10 | 10/10 |
| openvla_oft | libero_10/2 | on moka_pot_1 flat_stove_1_cook_region | 245, 264, 244, 262, 226, 240, 248, 239, 224, 248 | 0/10 | 10/10 |
| openvla_oft | libero_10/3 | close white_cabinet_1_bottom_region | 220, 226, 294, 212, 232, 192, 218, 211, 332, 201 | 0/10 | 10/10 |
| openvla_oft | libero_10/3 | in akita_black_bowl_1 white_cabinet_1_bottom_region | 144, 135, 144, 139, 151, 119, 144, 136, 187, 137 | 0/10 | 10/10 |
| openvla_oft | libero_10/4 | on porcelain_mug_1 plate_1 | 103, 90, 99, 116, 99, 112, 95, 117, 95, 107 | 0/10 | 10/10 |
| openvla_oft | libero_10/4 | on white_yellow_mug_1 plate_2 | 218, 210, 220, 237, 240, 227, 220, 229, 214, 224 | 0/10 | 10/10 |
| openvla_oft | libero_10/9 | in white_yellow_mug_1 microwave_1_heating_region | 150, 135, 145, 149, 148, 170, 139, 142, 141, 150 | 0/10 | 10/10 |
| openvla_oft | libero_10/9 | close microwave_1 | 259, 244, 246, 249, 283, 278, 244, 246, 251, 262 | 0/10 | 10/10 |
| openvla_oft | libero_spatial/0 | on akita_black_bowl_1 plate_1 | 75, 80, 84, 81, 78, 78, 74, 87, 75, 78 | 0/10 | 10/10 |
| openvla_oft | libero_spatial/1 | on akita_black_bowl_1 plate_1 | 105, 99, 109, 104, 107, 104, 106, 104, 94, 109 | 0/10 | 10/10 |

The only observed timeout is OpenPI libero_10/2 init 7 (seed 49), length 520: `turnon flat_stove_1` first observed true at step 78, remains true; `on moka_pot_1 flat_stove_1_cook_region` was never observed true and is false at timeout.
