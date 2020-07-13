

import Augmentor

p = Augmentor.Pipeline("/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/data_imgs/rgb_train_data_imgs")
p.ground_truth("/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/data_imgs/rgb_train_target_imgs") # has to have the same name with the input data
p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)

p.sample(5000)




p = Augmentor.Pipeline("/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/data_imgs/rgb_test_data_imgs")
p.ground_truth("/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/data_imgs/rgb_test_target_imgs") # has to have the same name with the input data
p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)

p.sample(1000)