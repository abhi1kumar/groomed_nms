# Groomd NMS variations before/after NMS
python plot/plot_prediction_with_ground_truth.py -i\
   output/groumd_nms/results/results_test_with_pred_no_nms/\
   output/groumd_nms/results/results_test_with_pred/\
   output/groumd_nms/results/results_test_with_pred_times_class_no_nms/\
   output/groumd_nms/results/results_test_with_pred_times_class/\
  --labels\
   GrooMeD-NMS\(Before\)\(Pred\)\
   GrooMeD-NMS\(After\)\(Pred\)\
   GrooMeD-NMS\(Before\)\(Pred*Class\)\
   GrooMeD-NMS\(After\)\(Pred*Class\)\
  --save_prefix before_after_nms

# Accept prob before/after NMS
python plot/plot_prediction_with_ground_truth.py -i\
    output/accept_prob_dynamic/results/results_test_with_pred_no_nms/\
    output/groumd_nms/results/results_test_with_pred_no_nms/\
    output/accept_prob_dynamic/results/results_test_with_pred/\
    output/groumd_nms/results/results_test_with_pred/\
  --labels\
    Without\(Before\)\(Pred\)\
    With\(Before\)\(Pred\)\
    Without\(After\)\(Pred\)\
    With\(After\)\(Pred\)\
  --save_prefix with_without

# Comparison across models
python plot/plot_prediction_with_ground_truth.py -i\
  output/M3D-RPN/results/results_test/\
  output/kitti_3d_uncertainty/results/results_test\
  output/Kinematic_video/results/results_test/\
  output/groumd_nms/results/results_test_with_pred_times_class/\
  --labels\
  M3D-RPN\
  Kinematic\ \(Image\)\
  Kinematic\ \(Video\)\
  GrooMeD-NMS


# Comparison across models on visibility 1
python plot/compare_performance_on_vis.py -i\
  output/M3D-RPN/results/results_test/\
  output/kitti_3d_uncertainty/results/results_test\
  output/Kinematic_video/results/results_test/\
  output/groumd_nms/results/results_test_with_pred_times_class/\
  --labels\
  M3D-RPN\
  Kinematic\ \(Image\)\
  Kinematic\ \(Video\)\
  GrooMeD-NMS