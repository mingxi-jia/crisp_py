import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node

def generate_launch_description():
    img_width, img_height = 480, 270
    frontview_w, frontview_h = 848, 480

    # The side view's depth stream is 12.2 MB/s of USB bandwidth and a
    # depth-to-colour alignment pass in the node, and pi0.5 never reads it --
    # its observation is two RGB images plus joint positions. Disabling it also
    # removes the "Depth stream start failure" hardware error this camera keeps
    # hitting. Set True if you need cam1 depth (point clouds, recording).
    frontview_depth = False
    fps = 15
    auto_parameter = True
    enable_pointcloud = False  # Set to False if you don't need pointcloud data
    return LaunchDescription([
        # Camera nodes
        GroupAction([
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='cam1',
                namespace='/',
                parameters=[{
                    'serial_no': '239222303046',
                    'enable_depth': frontview_depth,
                    'depth_module.depth_profile': f'{frontview_w}x{frontview_h}x{fps}',
                    'rgb_camera.color_profile': f'{frontview_w}x{frontview_h}x{fps}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'rgb_camera.enable_auto_exposure': auto_parameter,
                    'depth_module.enable_auto_exposure': auto_parameter,
                    'rgb_camera.enable_auto_white_balance': auto_parameter,
                    'rgb_camera.white_balance':3293.0,
                    'rgb_camera.exposure':100,
                    'enable_infra': False, 
                    'align_depth.enable': frontview_depth,
                    'initial_reset': True,
                    'camera_name': 'cam1',
                    'pointcloud.enable': enable_pointcloud,
                }]
            ),
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='cam4',
                namespace='/',
                parameters=[{
                    'serial_no': '218722271574',
                    # D405 has no separate RGB sensor: color and depth share the
                    # depth module, so both profiles must match and both live
                    # under depth_module.* (rgb_camera.* does not exist here).
                    'depth_module.depth_profile': f'{img_width}x{img_height}x{fps}',
                    'depth_module.color_profile': f'{img_width}x{img_height}x{fps}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'depth_module.enable_auto_exposure':False,
                    'depth_module.exposure':40557,
                    'depth_module.brightness':0,
                    'align_depth.enable': True,
                    'initial_reset': False,
                    'camera_name': 'cam4',
                }]
            ),
        ]),

        # Static transforms for each camera

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='bob_link_broadcaster',
            arguments=['0.61482', '0.523489', '0.638426', '0.321738', '0.30324', '-0.611973', '0.655759', 'fr3_link0', 'cam1_link']
            # arguments=['0.55482', '0.483489', '0.659426', '0.321738', '0.30324', '-0.611973', '0.655759', 'fr3_link0', 'cam1_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='mel_link_broadcaster',
            arguments=['0.508074', '-0.522051', '0.682903', '-0.330997', '0.343186', '0.628994', '0.614029', 'fr3_link0', 'cam2_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dave_link_broadcaster',
            arguments=['1.260047', '-0.08', '0.625', '-0.300660', '-0.006615', '0.953625', '0.012632', 'fr3_link0', 'cam3_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='workspace_link_broadcaster',
            arguments=['0.5', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', 'fr3_link0', 'workspace_link']
        ),
    ])
